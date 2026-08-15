"""Guard: a documented environment variable must actually reach its container.

Setting a variable in `.env` that compose never forwards is worse than not
documenting it. The variable silently does nothing, and every conclusion drawn
while "changing" it is wrong.

That is not hypothetical here. Eleven knobs were documented in `.env.example`
and read by service code but absent from every compose `environment:` block —
including ``WHISPER_COMPUTE_TYPE``, the one variable
``benchmarks/run_german_eval.py`` tells you to change when A/B-ing float16
against int8_float16. The benchmark would have transcribed with the same
configuration twice and reported "no significant difference" with full
confidence.

Compose does not forward the host environment by default: there is no
``env_file`` here, so a variable reaches a container only if it is named in that
service's ``environment:`` mapping.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml", reason="PyYAML needed to parse compose files")

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPOSE = REPO_ROOT / "docker-compose.yml"
ENV_EXAMPLE = REPO_ROOT / ".env.example"

# Variables that are documented but deliberately NOT in the base compose file.
# Each needs a reason; "we forgot" is not one.
EXEMPT: dict[str, str] = {
    # Host-side only: consumed by compose itself for interpolation and volume
    # placement, never needed inside a container.
    "APP_DATA_DIR": "host-side interpolation only",
    "IMAGE_REGISTRY": "host-side interpolation only",
    "IMAGE_TAG": "host-side interpolation only",
    "ENVIRONMENT": "host-side marker, not read by service code",
    # Set by the ROCm overlay's x-rocm-env, not by the CUDA base file.
    "HIP_VISIBLE_DEVICES": "set by docker-compose.rocm.yml",
    "ROCR_VISIBLE_DEVICES": "set by docker-compose.rocm.yml",
    "HSA_OVERRIDE_GFX_VERSION": "set by docker-compose.rocm.yml",
    # Port mappings: consumed by the ports: section, not by the app.
    **{f"{name}_PORT": "compose ports mapping" for name in (
        "FRONTEND", "PIPER_TTS", "STT", "QWEN3_ASR", "WHISPER_CPP",
        "QWEN3_TTS", "PARAKEET_ASR", "CANARY_ASR", "CHATTERBOX_TTS", "TRAINING",
    )},
}

# Directory overrides are wired through volume mounts rather than environment.
_DIR_SUFFIXES = ("_DIR",)


def _documented() -> set[str]:
    """Uncommented assignments in .env.example."""
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    return set(re.findall(r"^([A-Z0-9_]+)=", text, flags=re.M))


def _consumed_by_service() -> dict[str, set[str]]:
    """{service_dir_name: variables its Python reads}.

    Two shapes, because the second one hid the exact class of bug this file
    exists to catch. `ttl_from_env(os.getenv, "STT_MODEL_TTL", "MODEL_TTL")`
    passes `os.getenv` as a *callable* and the names as later arguments, so the
    direct-call pattern below never matched it — and the idle-unload TTLs, which
    the deployment-parity section calls "the whole VRAM story on a 12 GB card",
    were the one family of knobs this guard could not see.
    """
    out: dict[str, set[str]] = {}
    for path in REPO_ROOT.glob("*-service/*.py"):
        source = path.read_text(encoding="utf-8", errors="ignore")
        names = set(re.findall(
            r"os\.(?:getenv|environ\.get)\(\s*[\"']([A-Z0-9_]+)[\"']", source
        ))
        for call in re.findall(
            r"ttl_from_env\(\s*os\.(?:getenv|environ\.get)\s*,([^)]*)\)", source
        ):
            names.update(re.findall(r"[\"']([A-Z0-9_]+)[\"']", call))
        out.setdefault(path.parent.name, set()).update(names)
    return out


def _compose_environment() -> dict[str, set[str]]:
    """{service_name: keys in its environment: mapping}."""
    document = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    out: dict[str, set[str]] = {}
    for name, service in (document.get("services") or {}).items():
        env = service.get("environment") or {}
        if isinstance(env, dict):
            out[name] = set(env.keys())
        else:  # list form: ["KEY=value", ...]
            out[name] = {entry.split("=", 1)[0] for entry in env}
    return out


def test_documented_variables_reach_their_container():
    """Every documented var a service reads must be in that service's environment."""
    documented = _documented()
    consumed = _consumed_by_service()
    compose_env = _compose_environment()

    gaps: list[str] = []
    for service_dir, variables in sorted(consumed.items()):
        passed = compose_env.get(service_dir, set())
        for variable in sorted(variables):
            if variable not in documented:
                continue  # undocumented internals are out of scope here
            if variable in EXEMPT or variable.endswith(_DIR_SUFFIXES):
                continue
            if variable not in passed:
                gaps.append(f"{service_dir}: {variable}")

    assert not gaps, (
        "these variables are documented in .env.example and read by service "
        "code, but compose never forwards them — setting them does nothing:\n  "
        + "\n  ".join(gaps)
        + "\nAdd them to the service's environment: block, or add an EXEMPT "
          "entry here explaining why they belong somewhere else."
    )


def test_compute_type_is_wired_because_the_benchmark_depends_on_it():
    """Named explicitly: silently dropping this invalidates every A/B result."""
    passed = _compose_environment().get("stt-service", set())
    assert "WHISPER_COMPUTE_TYPE" in passed, (
        "benchmarks/run_german_eval.py A/Bs float16 against int8_float16 by "
        "setting WHISPER_COMPUTE_TYPE. If compose does not forward it, both runs "
        "use the same compute type and the comparison reports a confident "
        "'no significant difference' that means nothing."
    )


# --- the other direction ---------------------------------------------------
#
# The test above catches "documented and read, but never forwarded". It cannot
# see the mirror image: a variable compose *does* forward that nothing reads.
# That reads as a working knob in every deployment file and does nothing at all.
#
# Three were live when this was added. `STT_SERVICE_URL` on the training service
# was the worst of them: compose set it, the service ignored it, and took the
# address from a form field on /train instead — so the documented setting was
# inert and the value came from the request. `PIPER_DATA_DIR` was set next to
# `PIPER_OUTPUT_DIR`, but only the output half was read; every model path was the
# literal /app/models. `WORKERS: 1` looked like a uvicorn worker count while
# start.sh hardcodes `--workers 1`, so raising it would have done nothing (which
# was lucky — four workers means four copies of the model in VRAM).

# Consumed by something other than the service's own Python: the CUDA runtime,
# the PyTorch allocator, OpenMP, or the container entrypoint script.
NOT_READ_BY_PYTHON = {
    "CUDA_VISIBLE_DEVICES": "read by the CUDA runtime",
    "NVIDIA_VISIBLE_DEVICES": "read by the NVIDIA container toolkit",
    "NVIDIA_DRIVER_CAPABILITIES": "read by the NVIDIA container toolkit",
    "PYTORCH_CUDA_ALLOC_CONF": "read by the PyTorch caching allocator",
    "PYTORCH_JIT": "read by torch at import",
    "OMP_NUM_THREADS": "read by OpenMP / BLAS",
    "PYTHONUNBUFFERED": "read by CPython",
    "HIP_VISIBLE_DEVICES": "read by the ROCm runtime",
    "ROCR_VISIBLE_DEVICES": "read by the ROCm runtime",
    "HSA_OVERRIDE_GFX_VERSION": "read by the ROCm runtime",
    "WS_MAX_QUEUE": "read by stt-service/start.sh",
    "KEEPALIVE_TIMEOUT": "read by stt-service/start.sh",
    "WHISPER_MODEL": "read by whisper-cpp's entrypoint",
    "EXTRA_ARGS": "read by whisper-cpp's entrypoint",
}


def _shell_read_variables() -> set[str]:
    """Variables referenced by any service's entrypoint script."""
    names: set[str] = set()
    for script in REPO_ROOT.glob("*-service/*.sh"):
        names.update(re.findall(
            r"\$\{([A-Z0-9_]+)[:}]", script.read_text(encoding="utf-8")))
    return names


def test_no_service_is_handed_a_variable_nothing_reads():
    """Every key in a service's `environment:` must reach some consumer."""
    compose_env = _compose_environment()
    consumed = _consumed_by_service()
    shell_read = _shell_read_variables()

    dead: list[str] = []
    for service, keys in sorted(compose_env.items()):
        if not (REPO_ROOT / service / "app.py").exists():
            continue  # third-party image; its own entrypoint decides
        for key in sorted(keys):
            if key in NOT_READ_BY_PYTHON or key in shell_read:
                continue
            if key not in consumed.get(service, set()):
                dead.append(f"{service}: {key}")

    assert not dead, (
        "compose passes these to a container whose code never reads them, so "
        "setting them changes nothing:\n  " + "\n  ".join(dead)
        + "\nEither read the variable, delete it from the environment: block, or "
          "add it to NOT_READ_BY_PYTHON with the consumer that does read it."
    )


def test_not_read_by_python_entries_are_all_still_in_use():
    """An exemption for a variable nothing sets any more is dead weight.

    Scans every compose variant, not just the base file: the ROCm overlay is
    where the HIP/HSA knobs live, and `PYTORCH_JIT` is a Dockerfile ENV.
    """
    # Compose's own tags (`devices: !reset []` in the ROCm overlay) are not YAML
    # the SafeLoader knows. Only the environment mappings matter here, so unknown
    # tags resolve to their untagged value rather than failing the parse.
    class _ComposeLoader(yaml.SafeLoader):
        pass

    _ComposeLoader.add_multi_constructor(
        "", lambda loader, suffix, node: loader.construct_object(
            node.__class__(loader.DEFAULT_MAPPING_TAG if isinstance(node, yaml.MappingNode)
                           else loader.DEFAULT_SEQUENCE_TAG if isinstance(node, yaml.SequenceNode)
                           else loader.DEFAULT_SCALAR_TAG, node.value), deep=True)
    )

    all_keys: set[str] = set()
    for compose in REPO_ROOT.glob("docker-compose*.yml"):
        document = yaml.load(compose.read_text(encoding="utf-8"), Loader=_ComposeLoader) or {}
        for service in (document.get("services") or {}).values():
            env = (service or {}).get("environment") or {}
            all_keys |= set(env) if isinstance(env, dict) else {
                entry.split("=", 1)[0] for entry in env}
        # x-* extension blocks carry the ROCm overlay's shared env mapping.
        for key, value in document.items():
            if key.startswith("x-") and isinstance(value, dict):
                all_keys |= set(value)
    for dockerfile in REPO_ROOT.glob("*-service/Dockerfile*"):
        text = dockerfile.read_text(encoding="utf-8").replace("\\\n", " ")
        all_keys |= set(re.findall(r"^\s*ENV\s+([A-Z0-9_]+)=", text, flags=re.M))

    stale = sorted(name for name in NOT_READ_BY_PYTHON if name not in all_keys)
    assert not stale, (
        f"NOT_READ_BY_PYTHON lists variables nothing sets any more: {stale}"
    )


def test_the_training_service_reads_its_stt_url_from_the_environment():
    """Named explicitly: this one was inert while a form field decided instead.

    A caller could point the service at any host, have it POST the upload there,
    and read the connection error back out of the job status.
    """
    source = (REPO_ROOT / "piper-training-service" / "app.py").read_text(encoding="utf-8")
    assert 'os.getenv("STT_SERVICE_URL"' in source, (
        "piper-training-service does not read STT_SERVICE_URL, so the address "
        "compose configures is ignored"
    )
    assert "def resolve_stt_service_url" in source, (
        "the per-request stt_service_url field is no longer validated against "
        "the configured URL"
    )


def test_exempt_entries_are_still_documented():
    """An exemption for a variable nobody documents is dead weight."""
    documented = _documented()
    stale = [
        name for name in EXEMPT
        if name not in documented and not name.endswith("_PORT")
    ]
    assert not stale, (
        f"EXEMPT lists variables that .env.example no longer documents: {stale}. "
        f"Remove them so the exemption list stays meaningful."
    )


def test_every_compose_default_matches_the_documented_default():
    """`${VAR:-x}` in compose and `VAR=y` in .env.example must not disagree.

    Two different defaults for one knob means the documented value is a lie for
    anyone who has not copied .env.example to .env — which is the default state
    of a fresh checkout.
    """
    compose_text = COMPOSE.read_text(encoding="utf-8")
    example_text = ENV_EXAMPLE.read_text(encoding="utf-8")

    documented_values = dict(
        re.findall(r"^([A-Z0-9_]+)=(.*)$", example_text, flags=re.M)
    )
    # ${VAR:-default} — only the two-part form carries a default worth checking.
    compose_defaults = re.findall(r"\$\{([A-Z0-9_]+):-([^}]*)\}", compose_text)

    conflicts: list[str] = []
    for name, compose_default in compose_defaults:
        if name not in documented_values:
            continue
        documented_default = documented_values[name].strip().strip('"').strip("'")
        if compose_default.strip() != documented_default:
            conflicts.append(
                f"{name}: compose='{compose_default}' but .env.example='{documented_default}'"
            )

    assert not conflicts, (
        "compose and .env.example disagree on default values:\n  "
        + "\n  ".join(conflicts)
    )


# --- deployment parity -------------------------------------------------------
#
# docker-compose.truenas-app.yml is a STANDALONE file: TrueNAS users paste it
# into "Install via YAML", so it never merges with docker-compose.yml and gets
# none of its environment. It had drifted 21 keys behind — including
# PIPER_STRICT_LANGUAGE (whether a missing German voice 400s or silently returns
# English), both idle-unload TTLs (the whole VRAM story on a 12 GB card) and
# WHISPER_COMPUTE_TYPE. A TrueNAS deployment therefore behaved differently from
# a compose deployment of the same commit, and nothing said so.

TRUENAS_APP = REPO_ROOT / "docker-compose.truenas-app.yml"

# Keys that legitimately differ: the standalone file pins a single GPU and
# cannot build from source.
PARITY_IGNORE = {"CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "FORCE_ACCELERATION"}


def _env_of(document: dict, service: str) -> set[str]:
    env = ((document.get("services") or {}).get(service, {}) or {}).get("environment") or {}
    if isinstance(env, dict):
        return set(env)
    return {entry.split("=", 1)[0] for entry in env}


def test_truenas_app_matches_the_base_stack():
    """Every env key the base sets must also be set in the standalone file.

    Extra keys there are fine — it pins a GPU the base leaves flexible. Missing
    keys are not: they silently split the fleet.
    """
    base = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    standalone = yaml.safe_load(TRUENAS_APP.read_text(encoding="utf-8"))

    gaps: list[str] = []
    for service in sorted((standalone.get("services") or {})):
        if service not in (base.get("services") or {}):
            continue
        missing = _env_of(base, service) - _env_of(standalone, service) - PARITY_IGNORE
        for key in sorted(missing):
            gaps.append(f"{service}: {key}")

    assert not gaps, (
        "docker-compose.truenas-app.yml is missing environment keys the base "
        "stack sets, so a TrueNAS install behaves differently from a compose "
        "install of the same commit:\n  " + "\n  ".join(gaps)
    )
