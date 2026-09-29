"""Guards for the device presets in ``deploy/profiles/*.env``.

A preset is a promise: "run this command with this file and you get a working
stack on this device". Three ways it was silently false:

* ``rk3588.env`` and ``strixhalo.env`` enable whisper.cpp but never set
  ``DEFAULT_STT_PROVIDER``. The gateway then preselects ``whisper``, i.e.
  stt-service, which is CUDA-only and not started on those devices.
* ``rk3588.env`` set ``STT_MODEL_TTL`` / ``TTS_MODEL_TTL``: knobs of Python GPU
  services that cannot run on that board. ``strixhalo.env`` set
  ``GPU_MEMORY_BUDGET_GB``, which nothing in the repository has ever read.
* ``PIPER_DEFAULT_VOICE`` and ``FRONTEND_WORKERS`` sat in presets for a long time
  while no code or compose file consumed them.

Each preset documents its own launch command in its header. The checks below run
against exactly that command: which compose files, which ``--profile`` flags.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from compose_helpers import REPO_ROOT, dotenv, env_mapping, interpolate, load_compose
from frontend_loader import load_frontend_app

PROFILE_DIR = REPO_ROOT / "deploy" / "profiles"
PROFILES = sorted(PROFILE_DIR.glob("*.env"))
FRONTEND = "frontend-service"


def _ids(paths):
    return [p.stem for p in paths]


# --- what a preset launches -----------------------------------------------------------


def launch_command(profile: Path) -> tuple[list[str], list[str]]:
    """(compose files, --profile names) from the first `docker compose` line in the header."""
    lines = []
    for raw in profile.read_text(encoding="utf-8").splitlines():
        if not raw.startswith("#"):
            if lines:
                break
            continue
        lines.append(raw.lstrip("#").strip())

    commands: list[str] = []
    current: str | None = None
    for line in lines:
        if current is None:
            if not line.startswith("docker compose"):
                continue
            current = ""
        current = f"{current} {line.rstrip(chr(92)).strip()}".strip()
        if not line.endswith("\\"):
            commands.append(current)
            current = None

    for command in commands:
        tokens = shlex.split(command)
        files = [tokens[i + 1] for i, t in enumerate(tokens) if t == "-f"]
        profiles = [tokens[i + 1] for i, t in enumerate(tokens) if t == "--profile"]
        assert files and profiles, f"{profile.name}: cannot read files/profiles from {command!r}"
        return files, profiles
    raise AssertionError(f"{profile.name} documents no `docker compose ...` launch command")


def merged_services(files: list[str]) -> dict[str, dict]:
    """Services of the launch command; later files add to earlier ones, keys kept per file."""
    out: dict[str, dict] = {}
    for file in files:
        for name, service in (load_compose(REPO_ROOT / file).get("services") or {}).items():
            out.setdefault(name, {"_blobs": [], "profiles": None, "env": {}})
            entry = out[name]
            entry["_blobs"].append(json.dumps(service or {}, default=str))
            if (service or {}).get("profiles") is not None:
                entry["profiles"] = service["profiles"]
            entry["env"].update(env_mapping(service or {}))
    return out


def started(profile: Path) -> set[str]:
    files, selected = launch_command(profile)
    services = merged_services(files)
    names = set()
    for name, entry in services.items():
        service_profiles = entry["profiles"]
        if not service_profiles or set(service_profiles) & set(selected):
            names.add(name)
    return names


# --- the gateway's view of providers -----------------------------------------------------


@lru_cache(maxsize=None)
def _providers(flags: tuple[str, ...]) -> dict[str, dict]:
    env = {flag: "true" for flag in flags}
    return dict(load_frontend_app(env).PROVIDER_REGISTRY["providers"])


def _enable_flags() -> list[str]:
    return sorted(k for k in merged_services(["docker-compose.yml"])[FRONTEND]["env"]
                  if k.startswith("ENABLE_"))


@lru_cache(maxsize=None)
def provider_service_and_flag() -> dict[str, tuple[str, str | None]]:
    """{provider id: (compose service that serves it, ENABLE_* flag that shows it or None)}."""
    flags = tuple(_enable_flags())
    everything = _providers(flags)
    base = _providers(())
    flag_of: dict[str, str] = {}
    for flag in flags:
        for provider in set(_providers((flag,))) - set(base):
            flag_of[provider] = flag
    out = {}
    for provider_id, provider in everything.items():
        host = urlsplit(provider.get("internal_url") or "").hostname
        if host:
            out[provider_id] = (host, flag_of.get(provider_id))
    return out


def test_the_registry_maps_every_provider_to_a_compose_service():
    """The premise of the checks below: provider -> service comes from the gateway itself."""
    mapping = provider_service_and_flag()
    services = merged_services(["docker-compose.yml"])
    assert {"whisper", "whisper-cpp", "piper", "qwen3-asr", "parakeet", "canary"} <= set(mapping)
    unknown = {p: s for p, (s, _) in mapping.items() if s not in services}
    assert not unknown, f"providers pointing at services compose does not define: {unknown}"
    assert mapping["whisper-cpp"] == ("whisper-cpp", "ENABLE_WHISPER_CPP")
    assert mapping["whisper"][1] is None, "the default STT engine is not behind a flag"


# --- the presets -------------------------------------------------------------------------------


def test_there_are_presets_to_check():
    assert len(PROFILES) >= 4, PROFILES


@pytest.mark.parametrize("profile", PROFILES, ids=_ids(PROFILES))
def test_the_documented_launch_command_names_real_files_and_profiles(profile):
    files, selected = launch_command(profile)
    for file in files:
        assert (REPO_ROOT / file).is_file(), f"{profile.name}: -f {file} does not exist"
    known = set()
    for entry in merged_services(files).values():
        known |= set(entry["profiles"] or [])
    unknown = sorted(set(selected) - known)
    assert not unknown, f"{profile.name}: --profile {unknown} matches no service"
    assert started(profile), f"{profile.name} launches nothing"


@pytest.mark.parametrize("profile", PROFILES, ids=_ids(PROFILES))
def test_the_default_providers_are_providers_the_preset_actually_starts(profile):
    """The engine the UI preselects must be served by a container this preset starts."""
    env = dotenv(profile)
    files, _ = launch_command(profile)
    frontend_env = merged_services(files)[FRONTEND]["env"]
    running = started(profile)
    mapping = provider_service_and_flag()

    problems = []
    for key, kind in (("DEFAULT_STT_PROVIDER", "stt"), ("DEFAULT_TTS_PROVIDER", "tts")):
        effective = interpolate(str(frontend_env[key]), env)
        if effective not in mapping:
            problems.append(f"{key}={effective!r} is not a provider the gateway knows "
                            f"({sorted(mapping)})")
            continue
        service, flag = mapping[effective]
        if service not in running:
            problems.append(
                f"{key}={effective!r} is served by {service}, which "
                f"`{' '.join(launch_command(profile)[1])}` does not start "
                f"(set {key} explicitly for this device)")
        if flag and env.get(flag, "false") != "true":
            problems.append(f"{key}={effective!r} needs {flag}=true or the UI hides it")
    assert not problems, f"{profile.name}:\n  " + "\n  ".join(problems)


@pytest.mark.parametrize("profile", PROFILES, ids=_ids(PROFILES))
def test_optional_backends_are_started_and_shown_together(profile):
    """An ENABLE_* flag without its container is a permanently red indicator; a container
    without its flag is invisible in the UI."""
    env = dotenv(profile)
    running = started(profile)
    problems = []
    for provider, (service, flag) in provider_service_and_flag().items():
        if not flag:
            continue
        shown = env.get(flag, "false") == "true"
        if shown and service not in running:
            problems.append(f"{flag}=true but {service} is not started ({provider})")
        if service in running and not shown:
            problems.append(f"{service} is started but {flag} is not true, so the UI hides {provider}")
    assert not problems, f"{profile.name}:\n  " + "\n  ".join(problems)


def _consumers(name: str, services: dict[str, dict], running: set[str]) -> list[str]:
    """Started services whose compose definition refers to variable *name*."""
    pattern = re.compile(r"\$\{" + re.escape(name) + r"(?:[:}\-?+])|\$" + re.escape(name) + r"\b")
    found = []
    for service, entry in services.items():
        if service not in running:
            continue
        if name in entry["env"] or any(pattern.search(blob) for blob in entry["_blobs"]):
            found.append(service)
    return found


@pytest.mark.parametrize("profile", PROFILES, ids=_ids(PROFILES))
def test_every_variable_a_preset_sets_is_read_by_something_it_starts(profile):
    """A knob nobody reads looks like configuration and does nothing."""
    files, _ = launch_command(profile)
    services = merged_services(files)
    running = started(profile)

    dead = [name for name in dotenv(profile) if not _consumers(name, services, running)]
    assert not dead, (
        f"{profile.name} sets variables that no service it starts reads: {dead}.\n"
        "Remove them, or (if they belong to an optional overlay) leave them as comments."
    )


@pytest.mark.parametrize("profile", PROFILES, ids=_ids(PROFILES))
def test_presets_set_only_variables_the_project_documents(profile):
    """Typos in a preset (WHISPER_MODLE=...) are invisible; catch them against .env.example."""
    known = set(dotenv(REPO_ROOT / ".env.example"))
    # documented in comments of .env.example (build-time or overlay-only)
    text = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    known |= set(re.findall(r"^#\s*([A-Z][A-Z0-9_]{3,})=", text, flags=re.M))
    known |= set(re.findall(r"^#\s{2,}([A-Z][A-Z0-9_]{3,})\b", text, flags=re.M))
    # overlay-specific knobs documented in the overlay itself
    known |= {"GGML_VULKAN_DEVICE", "HSA_OVERRIDE_GFX_VERSION", "PYTORCH_ALLOC_CONF"}
    unknown = sorted(set(dotenv(profile)) - known)
    assert not unknown, f"{profile.name}: not documented in .env.example: {unknown}"


def test_variables_that_only_exist_for_the_gpu_services_are_not_set_on_devices_without_them():
    """The regression that motivated this file, stated directly."""
    for name in ("rk3588", "strixhalo"):
        env = dotenv(PROFILE_DIR / f"{name}.env")
        assert env.get("DEFAULT_STT_PROVIDER") == "whisper-cpp", name
        assert "GPU_MEMORY_BUDGET_GB" not in env
    rk3588 = dotenv(PROFILE_DIR / "rk3588.env")
    assert not {"STT_MODEL_TTL", "TTS_MODEL_TTL", "MODEL_TTL"} & set(rk3588)
    assert rk3588["FRONTEND_WORKERS"] == "1"


def test_piper_default_voice_in_the_presets_is_an_installable_voice():
    """A typo here would be accepted silently: the service falls back to any voice."""
    baked = re.search(r'^ARG PIPER_VOICES="([^"]+)"',
                      (REPO_ROOT / "piper-tts-service" / "Dockerfile").read_text(encoding="utf-8"),
                      flags=re.M).group(1).split()
    for profile in PROFILES:
        voice = dotenv(profile).get("PIPER_DEFAULT_VOICE")
        if voice:
            assert voice in baked, f"{profile.name}: {voice} is not one of the voices the image ships: {baked}"


# --- the compose CLI --------------------------------------------------------------------------


@pytest.mark.parametrize("profile", PROFILES, ids=_ids(PROFILES))
def test_the_documented_command_is_valid_compose(profile):
    """Runs the header's own command with `config -q` (no daemon needed)."""
    if not shutil.which("docker") or subprocess.run(
            ["docker", "compose", "version"], capture_output=True).returncode != 0:
        if os.environ.get("REQUIRE_DRIFT_SUITES") == "1":
            pytest.fail("the docker compose CLI is required in CI to validate the presets")
        pytest.skip("docker compose CLI not installed")
    files, selected = launch_command(profile)
    command = ["docker", "compose", "--env-file", str(profile.relative_to(REPO_ROOT))]
    for file in files:
        command += ["-f", file]
    for name in selected:
        command += ["--profile", name]
    result = subprocess.run([*command, "config", "-q"], cwd=REPO_ROOT, capture_output=True,
                            text=True, timeout=120)
    assert result.returncode == 0, result.stderr
