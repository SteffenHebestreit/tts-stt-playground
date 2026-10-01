"""What the NeMo images (parakeet, canary, magpie) install: NeMo line, torch pin, and the build-time torch check.

Background (nemo_toolkit 3.0.0, published 2026-08-07): ``nemo_toolkit[asr]>=2.0.0``
already resolves to it, and torch was unpinned, so a rebuild changed the NeMo
release, the torch release and the transitive stack (transformers 5,
huggingface_hub 1) without any file changing. The recipe now takes

* ``NEMO_TOOLKIT_SPEC``      (the nemo_toolkit line; ``requirements.txt`` carries the same default),
* ``TORCH_VERSION`` / ``TORCHAUDIO_VERSION`` (pinned to a release that has a CUDA 12.8 wheel),

and refuses to finish the build when the installed torch cannot run on an RTX 50
card. Nothing here builds an image (no daemon); the RUN commands that decide the
outcome are extracted from the Dockerfile and executed, against a stand-in
``torch`` for the check and against the real requirements file for the filter.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[1]
# Every cu128 NeMo image and the nemo_toolkit extra it installs. Each guard below runs
# for each of them, and test_every_nemo_3_image_is_in_the_table_and_nothing_else keeps
# this table and the Dockerfiles in step. test_dockerfile_recipes.py reads it too.
NEMO_EXTRA = {"parakeet-asr-service": "asr", "canary-asr-service": "asr", "magpie-tts-service": "tts"}
CUDA_IMAGES = [(service, REPO_ROOT / service / "Dockerfile") for service in NEMO_EXTRA]
ROCM_IMAGE = REPO_ROOT / "parakeet-asr-service" / "Dockerfile.rocm"


def _instructions(dockerfile: Path) -> list[tuple[str, str]]:
    """(INSTRUCTION, arguments) with continuations folded in and comments dropped."""
    text = dockerfile.read_text(encoding="utf-8").replace("\\\n", " ")
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        keyword, _, rest = line.partition(" ")
        out.append((keyword.upper(), rest.strip()))
    return out


def _arg_default(dockerfile: Path, name: str) -> str:
    for keyword, args in _instructions(dockerfile):
        if keyword == "ARG" and args.startswith(f"{name}="):
            return shlex.split(args.split("=", 1)[1])[0]
    raise AssertionError(f"{dockerfile.name} declares no ARG {name} with a default")


def _requirements(service: str) -> dict[str, Requirement]:
    out = {}
    for raw in (REPO_ROOT / service / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            requirement = Requirement(line)
            out[requirement.name.lower().replace("_", "-")] = requirement
    return out


def _run_lines(dockerfile: Path) -> list[str]:
    return [args for keyword, args in _instructions(dockerfile) if keyword == "RUN"]


def _torch_checks(dockerfile: Path) -> list[str]:
    """The ``python3 -c`` programs of the RUN instructions that inspect torch's build."""
    programs = []
    for run in _run_lines(dockerfile):
        argv = shlex.split(run)
        for i, token in enumerate(argv[:-1]):
            if token == "-c" and "_cuda_getArchFlags" in argv[i + 1]:
                programs.append(argv[i + 1])
    return programs


# --- NeMo line ------------------------------------------------------------------------------

@pytest.mark.parametrize("service, dockerfile", CUDA_IMAGES, ids=[s for s, _ in CUDA_IMAGES])
def test_requirements_and_dockerfile_default_to_the_same_nemo_line(service, dockerfile):
    from_requirements = _requirements(service)["nemo-toolkit"]
    default = _arg_default(dockerfile, "NEMO_TOOLKIT_SPEC")

    assert from_requirements.extras == {NEMO_EXTRA[service]}
    assert str(from_requirements.specifier) == str(Requirement(f"nemo_toolkit{default}").specifier), (
        "requirements.txt and the NEMO_TOOLKIT_SPEC default disagree; the build argument replaces the "
        "requirements line, so whichever is stale is the one that silently stops mattering"
    )


@pytest.mark.parametrize("service, dockerfile", CUDA_IMAGES, ids=[s for s, _ in CUDA_IMAGES])
def test_the_default_line_is_the_verified_3_0_release_and_excludes_neighbours(service, dockerfile):
    spec = Requirement(f"nemo_toolkit{_arg_default(dockerfile, 'NEMO_TOOLKIT_SPEC')}").specifier

    assert Version("3.0.0") in spec
    assert Version("3.0.1") in spec, "patch releases are the point of a range"
    assert Version("2.7.3") not in spec, "the 2.x line is the rollback, chosen with the build argument"
    assert Version("3.1.0") not in spec, "an unverified minor release must be an explicit decision"
    assert Version("4.0.0") not in spec


def test_the_rocm_image_stays_on_2x_because_its_torch_is_below_nemo_3s_floor():
    spec = Requirement(f"nemo_toolkit{_arg_default(ROCM_IMAGE, 'NEMO_TOOLKIT_SPEC')}").specifier

    assert Version("2.7.3") in spec
    assert Version("3.0.0") not in spec


@pytest.mark.parametrize(
    "dockerfile", [d for _, d in CUDA_IMAGES] + [ROCM_IMAGE], ids=lambda d: f"{d.parent.name}/{d.name}"
)
def test_the_build_argument_replaces_the_requirements_line_instead_of_conflicting_with_it(dockerfile, tmp_path):
    """Run the Dockerfile's own filter on the real requirements.txt."""
    service = dockerfile.parent.name
    extra = NEMO_EXTRA[service]
    install = next(run for run in _run_lines(dockerfile) if "requirements.app.txt" in run)
    filter_command = install.split("&&")[0].strip()
    assert filter_command.startswith("grep")
    (tmp_path / "requirements.txt").write_text(
        (dockerfile.parent / "requirements.txt").read_text(encoding="utf-8"), encoding="utf-8")

    subprocess.run(["sh", "-c", filter_command], cwd=tmp_path, check=True)

    kept = (tmp_path / "requirements.app.txt").read_text(encoding="utf-8")
    names = {Requirement(line.split("#", 1)[0].strip()).name for line in kept.splitlines()
             if line.split("#", 1)[0].strip()}
    normalized = {n.lower().replace("_", "-") for n in names}
    assert "nemo-toolkit" not in normalized, \
        "pip would see two nemo_toolkit requirements and fail to resolve the overridden one"
    assert normalized == set(_requirements(service)) - {"nemo-toolkit"}, \
        "the filter must drop the nemo_toolkit line and nothing else"
    assert {"fastapi", "soundfile", "numpy"} <= names
    if extra == "asr":
        assert "librosa" in names
    # the ARG is what supplies NeMo instead
    assert f"nemo_toolkit[{extra}]${{NEMO_TOOLKIT_SPEC}}" in install


# --- torch ----------------------------------------------------------------------------------

@pytest.mark.parametrize("service, dockerfile", CUDA_IMAGES, ids=[s for s, _ in CUDA_IMAGES])
def test_torch_and_torchaudio_are_pinned_to_the_same_release_from_the_cu128_index(service, dockerfile):
    torch_version = _arg_default(dockerfile, "TORCH_VERSION")
    assert Version(torch_version), "TORCH_VERSION must be a real release"
    assert _arg_default(dockerfile, "TORCHAUDIO_VERSION") == torch_version, \
        "torch and torchaudio are released in lockstep; a mismatch fails at import"
    assert Version(torch_version) >= Version("2.7"), "Blackwell (sm_120) needs a CUDA 12.8 build, 2.7 or later"

    install = next(run for run in _run_lines(dockerfile) if "download.pytorch.org/whl/" in run)
    argv = shlex.split(install)
    assert "--no-cache-dir" in argv
    assert "torch==${TORCH_VERSION}" in argv and "torchaudio==${TORCHAUDIO_VERSION}" in argv
    assert argv[argv.index("--index-url") + 1] == "https://download.pytorch.org/whl/cu128"


@pytest.mark.parametrize("service, dockerfile", CUDA_IMAGES, ids=[s for s, _ in CUDA_IMAGES])
def test_every_pip_install_skips_the_wheel_cache(service, dockerfile):
    installs = [run for run in _run_lines(dockerfile) if "pip3 install" in run and "--upgrade pip" not in run]
    assert len(installs) == 2
    assert all("--no-cache-dir" in run for run in installs)


@pytest.mark.parametrize("service, dockerfile", CUDA_IMAGES, ids=[s for s, _ in CUDA_IMAGES])
def test_torch_layer_precedes_the_nemo_layer_so_a_requirements_edit_keeps_the_torch_download(service, dockerfile):
    runs = _run_lines(dockerfile)
    torch_at = next(i for i, run in enumerate(runs) if "download.pytorch.org" in run)
    nemo_at = next(i for i, run in enumerate(runs) if "requirements.app.txt" in run)
    assert torch_at < nemo_at


# --- numpy / cuda-bindings ------------------------------------------------------------------

@pytest.mark.parametrize("service", [s for s, _ in CUDA_IMAGES])
def test_numpy_is_not_capped_below_2(service):
    """The cap came from NeMo 2.3; 2.7.3 and 3.0.0 both ran the ASR API on numpy 2.4."""
    numpy = _requirements(service)["numpy"].specifier
    assert Version("1.26.4") in numpy
    assert Version("2.4.6") in numpy


@pytest.mark.parametrize("service", [s for s, _ in CUDA_IMAGES])
def test_cuda_bindings_stay_on_the_cuda_12_generation_of_the_wheels_we_install(service):
    """NeMo's CUDA-graph decoder picks call signatures from this package's major version."""
    bindings = _requirements(service)["cuda-bindings"].specifier
    assert Version("12.9.4") in bindings  # what torch 2.10's CUDA 12.8 build pins
    assert Version("13.0.0") not in bindings


# --- the build-time torch check -------------------------------------------------------------

def _run_check(program: str, tmp_path: Path, *, cuda="12.8", flags="sm_75 sm_90 sm_120", arch_list=None):
    """Execute *program* against a stand-in ``torch``; returns (exit code, stdout, stderr)."""
    package = tmp_path / "torch"
    package.mkdir(exist_ok=True)
    (package / "__init__.py").write_text(textwrap.dedent(f"""
        from types import SimpleNamespace
        __version__ = "2.11.0+cu128"
        version = SimpleNamespace(cuda={cuda!r})
        _C = SimpleNamespace(_cuda_getArchFlags=lambda: {flags!r})
        # what the real torch.cuda.get_arch_list() answers on a host without a GPU, i.e. a docker build
        cuda = SimpleNamespace(get_arch_list=lambda: {list(arch_list or [])!r})
    """))
    done = subprocess.run(
        [sys.executable, "-c", program], cwd=tmp_path, env={"PYTHONPATH": str(tmp_path), "PATH": ""},
        capture_output=True, text=True, timeout=30,
    )
    return done.returncode, done.stdout, done.stderr


@pytest.mark.parametrize("service, dockerfile", CUDA_IMAGES, ids=[s for s, _ in CUDA_IMAGES])
def test_the_torch_check_runs_after_torch_is_installed_and_again_after_nemo(service, dockerfile):
    programs = _torch_checks(dockerfile)
    assert len(programs) == 2, "one check right after the torch install (fails fast), one after NeMo"
    assert programs[0] == programs[1]


@pytest.mark.parametrize("service, dockerfile", CUDA_IMAGES, ids=[s for s, _ in CUDA_IMAGES])
def test_a_cuda_12_8_build_with_blackwell_kernels_passes_even_though_the_build_host_has_no_gpu(
    service, dockerfile, tmp_path
):
    """Regression: torch.cuda.get_arch_list() is [] without a GPU, so a check built on it fails every build."""
    code, out, err = _run_check(_torch_checks(dockerfile)[0], tmp_path, arch_list=[])

    assert code == 0, err
    assert "sm_120" in out


@pytest.mark.parametrize("service, dockerfile", CUDA_IMAGES, ids=[s for s, _ in CUDA_IMAGES])
@pytest.mark.parametrize(
    "case, kwargs",
    [
        ("cuda 12.6 (2.12+ default on the cu126 index)", {"cuda": "12.6"}),
        ("cuda 13.0 (the PyPI default for torch 2.11+)", {"cuda": "13.0"}),
        ("no CUDA at all (a CPU wheel)", {"cuda": None}),
        ("no sm_120 kernels (cu128 but built without Blackwell)", {"flags": "sm_75 sm_80 sm_90"}),
        ("empty arch list", {"flags": ""}),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_a_torch_that_cannot_run_on_an_rtx_50_card_fails_the_build(service, dockerfile, tmp_path, case, kwargs):
    code, out, err = _run_check(_torch_checks(dockerfile)[0], tmp_path, **kwargs)

    assert code != 0, f"{case}: the build would have shipped it"
    assert "FATAL" in err and "sm_120" in err


# --- every image: the same trap ---------------------------------------------------------------

def _all_dockerfiles() -> list[Path]:
    return sorted(
        path for path in REPO_ROOT.glob("*/Dockerfile*")
        if path.is_file() and path.parent.name != "tests"
    )


@pytest.mark.parametrize("dockerfile", _all_dockerfiles(), ids=lambda d: f"{d.parent.name}/{d.name}")
def test_no_build_step_asks_torch_for_its_arch_list_through_the_gpu_api(dockerfile):
    """``torch.cuda.get_arch_list()`` returns ``[]`` unless a GPU is visible, and a build host has none.

    Checked on torch 2.11.0: ``if not is_available(): return []``. A ``RUN`` that asserts
    ``'sm_120' in torch.cuda.get_arch_list()`` therefore fails every build, on the day the
    image is first built. ``torch._C._cuda_getArchFlags()`` reads the compiled-in list
    without a device.
    """
    offenders = [run for run in _run_lines(dockerfile) if "get_arch_list" in run]
    assert not offenders, (
        f"{dockerfile.parent.name}/{dockerfile.name} checks the arch list with torch.cuda.get_arch_list(), "
        f"which is empty on a GPU-less build host, so the build would fail: use "
        f"torch._C._cuda_getArchFlags().split()"
    )


# --- Python: NeMo 3 unpacks .nemo files with tarfile's filter= argument ---------------------------------------

def _nemo3_dockerfiles() -> list[Path]:
    """Every image whose default NeMo line admits 3.x, so the next NeMo service is covered the day it appears."""
    found = []
    for path in _all_dockerfiles():
        if "nemo_toolkit" not in path.read_text(encoding="utf-8"):
            continue
        try:
            spec = Requirement(f"nemo_toolkit{_arg_default(path, 'NEMO_TOOLKIT_SPEC')}").specifier
        except AssertionError:
            continue
        if Version("3.0.0") in spec:
            found.append(path)
    return found


NEMO3_IMAGES = _nemo3_dockerfiles()
_NEMO3_IDS = lambda d: f"{d.parent.name}/{d.name}"  # noqa: E731


def _data_filter_checks(dockerfile: Path) -> list[tuple[int, str]]:
    """(index among the RUN steps, python program) for each RUN that tests for tarfile.data_filter."""
    checks = []
    for index, run in enumerate(_run_lines(dockerfile)):
        argv = shlex.split(run)
        for i, token in enumerate(argv[:-1]):
            if token == "-c" and "data_filter" in argv[i + 1]:
                checks.append((index, argv[i + 1]))
    return checks


def test_the_python_guard_covers_parakeet_and_canary_and_not_the_2x_rocm_image():
    names = {d.parent.name for d in NEMO3_IMAGES}
    assert {"parakeet-asr-service", "canary-asr-service"} <= names
    assert ROCM_IMAGE not in NEMO3_IMAGES, "the ROCm image stays on NeMo 2.x, which has no such requirement"


def test_every_nemo_3_image_is_in_the_table_and_nothing_else():
    """NEMO3_IMAGES is found by scanning the Dockerfiles, CUDA_IMAGES comes from NEMO_EXTRA.

    The guards in this file are split between the two lists, so an image missing from either
    one (a new NeMo service nobody added to the table, or a renamed NEMO_TOOLKIT_SPEC) would
    silently lose half of them. Magpie was missing from the hand-written list that preceded
    the table, so none of the requirements, torch and cuda-bindings guards ran for it.
    """
    assert set(NEMO3_IMAGES) == {dockerfile for _, dockerfile in CUDA_IMAGES}


@pytest.mark.parametrize("dockerfile", NEMO3_IMAGES, ids=_NEMO3_IDS)
def test_python_3_11_comes_from_deadsnakes_because_ubuntu_22_04s_is_a_release_candidate(dockerfile):
    """Ubuntu 22.04's python3.11 package is 3.11.0~rc1, which has no tarfile.data_filter.

    NeMo 3.0 passes ``filter="data"`` to ``TarFile.extract`` for every .nemo checkpoint it
    restores, so on the stock package every model load fails with a TypeError. The images
    built that way were published before anyone loaded a model in them.
    """
    runs = _run_lines(dockerfile)
    install = next(run for run in runs if "python3.11 " in run + " " and "apt-get install" in run)
    ppa = next((i for i, run in enumerate(runs) if "add-apt-repository" in run and "ppa:deadsnakes/ppa" in run), None)

    assert ppa is not None, f"{dockerfile.parent.name} installs Ubuntu's python3.11 (3.11.0~rc1), which NeMo 3 cannot run on"
    assert ppa <= runs.index(install)
    if runs[ppa] == install:
        assert install.index("ppa:deadsnakes/ppa") < install.index("python3.11")


@pytest.mark.parametrize("dockerfile", NEMO3_IMAGES, ids=_NEMO3_IDS)
def test_the_ppa_setup_cannot_stop_to_ask_for_a_time_zone(dockerfile):
    """The python3.11 install pulls in tzdata, which prompts "Geographic area:" unless told not to.

    An unattended build then waits on stdin until the CI job times out. The variable is set
    on each command rather than as ENV, so it does not end up in the running container.
    """
    setup = next(run for run in _run_lines(dockerfile) if "add-apt-repository" in run)

    assert setup.count("apt-get install") >= 2
    assert setup.count("apt-get install") == setup.count("DEBIAN_FRONTEND=noninteractive apt-get install")
    assert not any(k == "ENV" and "DEBIAN_FRONTEND" in args for k, args in _instructions(dockerfile)), \
        "as ENV it would stay set in the running container; scope it to the RUN"


@pytest.mark.parametrize("dockerfile", NEMO3_IMAGES, ids=_NEMO3_IDS)
def test_the_ppa_tooling_is_purged_before_python_3_11_and_pip_arrive(dockerfile):
    """software-properties-common is needed for add-apt-repository only.

    It brings about 25 of Ubuntu's Python 3.10 packages into /usr/lib/python3/dist-packages,
    which python3.11 has on its sys.path too: a distutils-era blinker 1.4 that pip cannot
    uninstall, six and more-itertools that pip takes instead of installing its own, and a
    pygobject that fails `pip check`. Purging it right after the PPA is added removes them
    before pip first runs. gpg-agent and ca-certificates stay (the CUDA base's gnupg2 depends
    on gpg-agent), and so do the PPA's list and key, which no package owns.
    """
    setup = next(run for run in _run_lines(dockerfile) if "add-apt-repository" in run)
    steps = [step.strip() for step in setup.split("&&")]
    ppa = next(i for i, step in enumerate(steps)
               if step.startswith("add-apt-repository") and "ppa:deadsnakes/ppa" in step)
    python = next(i for i, step in enumerate(steps) if "apt-get install" in step and "python3.11" in shlex.split(step))
    purges = [i for i, step in enumerate(steps) if "apt-get purge" in step]

    assert len(purges) == 1, f"{dockerfile.parent.name} keeps software-properties-common and its Python 3.10 packages"
    assert ppa < purges[0] < python, "purge once the PPA is added, and before python3.11 and pip are installed"
    argv = shlex.split(steps[purges[0]])
    assert argv[:3] == ["DEBIAN_FRONTEND=noninteractive", "apt-get", "purge"], \
        "a purge can stop to ask questions too; scope DEBIAN_FRONTEND to it like the installs"
    assert "-y" in argv
    assert "--auto-remove" in argv, "without it the Python 3.10 packages it pulled in stay behind"
    assert [a for a in argv[3:] if not a.startswith("-")] == ["software-properties-common"], \
        "only the PPA tool goes: the CUDA base's gnupg2 depends on gpg-agent, and TLS needs ca-certificates"


@pytest.mark.parametrize("dockerfile", NEMO3_IMAGES, ids=_NEMO3_IDS)
def test_the_build_fails_before_the_big_layers_when_python_cannot_unpack_a_nemo_file(dockerfile):
    checks = _data_filter_checks(dockerfile)
    assert len(checks) == 1, f"{dockerfile.parent.name} must check tarfile.data_filter exactly once"
    index, _ = checks[0]
    runs = _run_lines(dockerfile)

    assert index > next(i for i, r in enumerate(runs) if "ln -sf /usr/bin/python3.11 /usr/bin/python3" in r), \
        "the check must run against the python3 that pip and uvicorn use, so after the symlink"
    assert index < next(i for i, r in enumerate(runs) if "download.pytorch.org" in r), \
        "fail before the multi-GB torch layer, not after it"


@pytest.mark.parametrize("dockerfile", NEMO3_IMAGES, ids=_NEMO3_IDS)
def test_the_python_check_passes_with_data_filter_and_fails_the_build_without_it(dockerfile):
    import tarfile

    program = _data_filter_checks(dockerfile)[0][1]
    if not hasattr(tarfile, "data_filter"):
        pytest.skip("this interpreter is itself older than 3.11.4")

    ok = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, timeout=30)
    assert ok.returncode == 0, ok.stderr

    # the interpreter Ubuntu 22.04 ships: same module, without the attribute
    stripped = "import tarfile\ndel tarfile.data_filter\n" + program
    bad = subprocess.run([sys.executable, "-c", stripped], capture_output=True, text=True, timeout=30)
    assert bad.returncode != 0, "a Python without tarfile.data_filter would have shipped"
    assert "FATAL" in bad.stderr and "3.11.4" in bad.stderr and "NeMo 3" in bad.stderr
