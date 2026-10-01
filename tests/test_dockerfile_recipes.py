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
* D4  NeMo is installed with torch held by a constraint file (parakeet, canary, magpie), and
      no NeMo layer leaves pip's wheel cache in /root/.cache, where the model-cache volume is;
* D5  the Piper release asset name survives a builder that does not set TARGETARCH;
* D6  the Magpie build fails when the NeMo it installed cannot do what the service imports it
      for, and bakes in the Japanese dictionary pyopenjtalk would otherwise download at runtime.
"""

from __future__ import annotations

import ast
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
from test_nemo3_dependencies import NEMO_EXTRA  # the one table of NeMo images and their extras

STT = REPO_ROOT / "stt-service"
STT_CUDA, STT_ROCM = STT / "Dockerfile", STT / "Dockerfile.rocm"
NEMO_IMAGES = [REPO_ROOT / service / "Dockerfile" for service in NEMO_EXTRA]
MAGPIE = REPO_ROOT / "magpie-tts-service" / "Dockerfile"
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
    """Every package named by any `apt-get install` step.

    A Dockerfile may install the tools that add a PPA first and its real packages second
    (the NeMo images take python3.11 from deadsnakes), so the first step is not enough.
    """
    packages: list[str] = []
    for run in _runs(dockerfile):
        for step in run.split("apt-get install")[1:]:
            packages += [t for t in shlex.split(step.split("&&")[0]) if not t.startswith("-")]
    return packages


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
    assert "rm" not in shlex.split(command), \
        "this test runs the NeMo install RUN on the test host with only pip3 and python3 stubbed; put a cleanup in a RUN of its own"
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
    nemo = f"nemo_toolkit[{NEMO_EXTRA[dockerfile.parent.name]}]>=3.0.0,<3.1"
    assert argv[argv.index("-c") + 1] == str(constraints)
    assert constraints.read_text().split() == [f"torch=={torch}", f"torchaudio=={audio}"]
    assert nemo in argv, "the NeMo spec is unchanged"
    assert argv.index("-c") < argv.index(nemo), "-c must apply to the whole install"
    assert argv[argv.index("-r") + 1] == "requirements.app.txt"


@pytest.mark.parametrize("dockerfile", NEMO_IMAGES, ids=lambda p: p.parent.name)
def test_the_nemo_defaults_are_untouched(dockerfile):
    assert _arg_default(dockerfile, "NEMO_TOOLKIT_SPEC") == ">=3.0.0,<3.1"
    assert _arg_default(dockerfile, "TORCH_VERSION") == _arg_default(dockerfile, "TORCHAUDIO_VERSION") == "2.11.0"


@pytest.mark.parametrize("dockerfile", NEMO_IMAGES, ids=lambda p: p.parent.name)
def test_the_nemo_layer_leaves_no_pip_cache_in_the_model_cache_directory(dockerfile):
    """pip does not pass `--no-cache-dir` on to the pip subprocess that installs a source
    package's build requirements, but that subprocess inherits PIP_NO_CACHE_DIR. Magpie's NeMo
    layer kept 75 MB of downloads in /root/.cache/pip, and Docker copies an image's /root/.cache
    into every new model-cache volume mounted there.

    An ARG, so the variable is set for the RUN but not in the running container, and declared
    after the torch layer: an ARG changes the cache key of every RUN after it, and the torch
    download is the one worth keeping.
    """
    steps = _instructions(dockerfile)
    declared = [i for i, (k, args) in enumerate(steps) if k == "ARG" and args.split("=", 1)[0] == "PIP_NO_CACHE_DIR"]
    torch_at = next(i for i, (k, args) in enumerate(steps) if k == "RUN" and "download.pytorch.org/whl/" in args)
    nemo_at = next(i for i, (k, args) in enumerate(steps) if k == "RUN" and "requirements.app.txt" in args)

    assert len(declared) == 1, f"{dockerfile.parent.name}: the NeMo layer's source builds fill /root/.cache/pip"
    assert _arg_default(dockerfile, "PIP_NO_CACHE_DIR") == "1"
    assert torch_at < declared[0] < nemo_at
    assert not any(k == "ENV" and "PIP_NO_CACHE_DIR" in args for k, args in steps), \
        "as ENV it would stay set in the running container"


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


# --- D6: what the Magpie build proves before it ships ---------------------------------------------------------------
#
# CI cannot build a NeMo image, so these run the build-time programs against stand-in packages,
# never the RUN itself: each RUN ends in `rm -rf /root/.cache`.


def _build_time_checks(dockerfile: Path) -> list[str]:
    """The RUN steps after the NeMo install and before `COPY app.py`.

    They run whenever the NeMo layer is rebuilt, and an edit to the service code does not repeat them.
    They must be RUNs of their own: a test executes the NeMo install RUN on the test host.
    """
    steps = _instructions(dockerfile)
    nemo = next(i for i, (k, args) in enumerate(steps) if k == "RUN" and "requirements.app.txt" in args)
    app = next(i for i, (k, args) in enumerate(steps) if k == "COPY" and args.split()[0] == "app.py")
    assert nemo < app
    return [args for k, args in steps[nemo + 1:app] if k == "RUN"]


def _python_step(run: str) -> tuple[dict[str, str], str, list[str]]:
    """(the variables set for python3, its -c program, the commands after it) of a `RUN python3 -c ...`."""
    argv = shlex.split(run)
    at = argv.index("python3")
    assert argv[at + 1] == "-c"
    return dict(token.split("=", 1) for token in argv[:at]), argv[at + 2], argv[at + 3:]


def _magpie_check(marker: str) -> tuple[dict[str, str], str, list[str]]:
    runs = [run for run in _build_time_checks(MAGPIE) if marker in run]
    assert len(runs) == 1, f"the Magpie image needs one build-time check that uses {marker}, after the NeMo install"
    return _python_step(runs[0])


def _assert_the_check_decides_the_build_and_cleans_up(after: list[str]) -> None:
    """`check && rm -rf ...`: with `;` or `|| true` in between, the RUN's status would be the cleanup's."""
    assert after[:2] == ["&&", "rm"], f"the check must decide the RUN's exit status, not {' '.join(after)!r}"
    assert "/root/.cache" in after, "what it leaves in /root/.cache would be copied into every new model-cache volume"
    assert "||" not in after and ";" not in after


def _imports(source: str) -> set[tuple[str, str]]:
    """(module, name) for each `from module import name`, (module, "") for each `import module`."""
    found: set[tuple[str, str]] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            found |= {(node.module, alias.name) for alias in node.names}
        elif isinstance(node, ast.Import):
            found |= {(alias.name, "") for alias in node.names}
    return found


def _run_program(program: str, packages: Path):
    return subprocess.run(
        [sys.executable, "-c", program], cwd=packages, env={"PYTHONPATH": str(packages), "PATH": ""},
        capture_output=True, text=True, timeout=30, stdin=subprocess.PIPE,
    )


def test_the_magpie_build_imports_everything_the_service_imports_from_nemo():
    """NeMo 2.7.3 installs, imports and has MagpieTTSModel, but not LANGUAGE_TOKENIZER_MAP, which
    app.py imports while it loads the model: an image built on the 2.x line passed its build and
    failed its first request. NeMo also swallows a broken nemo_text_processing (pynini) and then
    drops every digit from the text, so the build imports the normalizer too.
    """
    env, program, after = _magpie_check("MagpieTTSModel")
    service: set[tuple[str, str]] = set()
    for module in sorted(MAGPIE.parent.glob("*.py")):
        service |= {pair for pair in _imports(module.read_text(encoding="utf-8")) if pair[0].split(".")[0] == "nemo"}
    checked = _imports(program)

    assert service, "the service imports nothing from NeMo any more, so this check checks nothing"
    assert service <= checked, f"the build does not try {sorted(service - checked)}, which the service imports"
    assert ("nemo_text_processing.text_normalization.normalize", "Normalizer") in checked
    assert env.get("PYTORCH_JIT") == "0", \
        "torch 2.11's TorchScript compiler segfaults on a NeMo import, and the image's ENV is set later"
    _assert_the_check_decides_the_build_and_cleans_up(after)


def _stand_in_nemo(root: Path, *, nemo_3: bool = True, magpie: bool = True, pynini: bool = True) -> None:
    files = {
        "nemo/collections/tts/models/__init__.py": "class MagpieTTSModel:\n    pass\n" if magpie else "",
        # NeMo 2.7.3 has this module, but not the map
        "nemo/collections/tts/parts/utils/tts_dataset_utils.py":
            "LANGUAGE_TOKENIZER_MAP = {'de': ['german_phoneme']}\n" if nemo_3 else "def stack_tensors():\n    pass\n",
        "nemo_text_processing/text_normalization/normalize.py": "import pynini\n\n\nclass Normalizer:\n    pass\n",
    }
    if pynini:
        files["pynini.py"] = ""
    for name, body in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(body, encoding="utf-8")
    for directory in [path for path in root.rglob("*") if path.is_dir()]:
        (directory / "__init__.py").touch()


@pytest.mark.parametrize("kwargs, error", [
    pytest.param({}, None, id="nemo 3.0"),
    pytest.param({"nemo_3": False}, "LANGUAGE_TOKENIZER_MAP", id="nemo 2.7.3"),
    pytest.param({"magpie": False}, "MagpieTTSModel", id="a nemo without magpie"),
    pytest.param({"pynini": False}, "pynini", id="nemo_text_processing without a working pynini"),
])
def test_the_magpie_import_check_passes_on_nemo_3_and_fails_the_build_otherwise(tmp_path, kwargs, error):
    _, program, _ = _magpie_check("MagpieTTSModel")
    _stand_in_nemo(tmp_path, **kwargs)

    done = _run_program(program, tmp_path)

    if error is None:
        assert done.returncode == 0, done.stderr
    else:
        assert done.returncode != 0, "the image would have shipped"
        assert error in done.stderr


def test_the_magpie_build_bakes_in_the_openjtalk_dictionary():
    """pyopenjtalk 0.4.1, which nemo_toolkit[tts] pulls in for Japanese, downloads its 22.6 MB
    dictionary on the first g2p call. In the service that was the first Japanese request: under
    the single-flight model lock, through an urlopen with no timeout, into the container layer
    (lost with every recreation), and never on a host without egress. Called at build time, it
    puts the dictionary into pyopenjtalk's package directory, in the image.
    """
    env, program, after = _magpie_check("pyopenjtalk")

    assert "pyopenjtalk.g2p(" in program and "OPEN_JTALK_DICT_DIR" in program
    assert program.isascii(), "spell the katakana with escapes, so the Dockerfile stays ASCII"
    assert "OPEN_JTALK_DICT_DIR" not in env and not any(
        k == "ENV" and "OPEN_JTALK_DICT_DIR" in args for k, args in _instructions(MAGPIE)), \
        "the dictionary belongs in the image, not under /root/.cache, which the TrueNAS bind mount hides"
    _assert_the_check_decides_the_build_and_cleans_up(after)


def _stand_in_pyopenjtalk(root: Path, *, phonemes: str, dictionary: bool) -> None:
    dictionary_dir = root / "open_jtalk_dic_utf_8-1.11"
    if dictionary:
        dictionary_dir.mkdir()
    (root / "pyopenjtalk.py").write_text(textwrap.dedent(f"""
        # pyopenjtalk 0.4.1 keeps the path as bytes, and fetches the dictionary on the first g2p call
        OPEN_JTALK_DICT_DIR = {str(dictionary_dir).encode("utf-8")!r}

        def g2p(text):
            # phonemes for Japanese (kana, kanji); nothing for, say, an escape that reached Python unprocessed
            japanese = bool(text) and all(0x3040 <= ord(c) <= 0x30FF or 0x4E00 <= ord(c) <= 0x9FFF for c in text)
            return {phonemes!r} if japanese else ""
    """), encoding="utf-8")


@pytest.mark.parametrize("phonemes, dictionary, error", [
    pytest.param("t e s u t o", True, None, id="dictionary and g2p work"),
    pytest.param("", True, "g2p returned nothing", id="g2p returns nothing"),
    pytest.param("t e s u t o", False, "no OpenJTalk dictionary", id="no dictionary where pyopenjtalk looks"),
])
def test_the_dictionary_check_fails_the_build_without_a_working_dictionary(tmp_path, phonemes, dictionary, error):
    _, program, _ = _magpie_check("pyopenjtalk")
    _stand_in_pyopenjtalk(tmp_path, phonemes=phonemes, dictionary=dictionary)

    done = _run_program(program, tmp_path)

    if error is None:
        assert done.returncode == 0, done.stderr
    else:
        assert done.returncode != 0, "the image would have shipped"
        assert error in done.stderr
