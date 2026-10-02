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

from compose_helpers import env_mapping, interpolate, load_compose

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
    # ROCR_VISIBLE_DEVICES is deliberately NOT here: nothing sets it, because
    # the ROCr and HIP filters nest and forwarding both breaks any index above
    # 0. .env.example documents it only as something not to set.
    "HIP_VISIBLE_DEVICES": "set by docker-compose.rocm.yml",
    "HSA_OVERRIDE_GFX_VERSION": "set by docker-compose.rocm.yml",
    # Port mappings: consumed by the ports: section, not by the app.
    **{f"{name}_PORT": "compose ports mapping" for name in (
        "FRONTEND", "PIPER_TTS", "STT", "QWEN3_ASR", "WHISPER_CPP",
        "QWEN3_TTS", "PARAKEET_ASR", "CANARY_ASR", "CHATTERBOX_TTS", "MAGPIE_TTS", "TRAINING",
    )},
}

# Directory overrides are wired through volume mounts rather than environment.
_DIR_SUFFIXES = ("_DIR",)


def _documented() -> set[str]:
    """Uncommented assignments in .env.example."""
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    return set(re.findall(r"^([A-Z0-9_]+)=", text, flags=re.M))


# A read of an environment variable by name. Beyond the direct
# `os.getenv("X")` / `os.environ.get("X")` this covers the small wrappers the
# services use to parse numbers and flags (`env_number("X", ...)`,
# `_env_number`, `env_int`, `env_flag`): any callable with "env" in its name whose
# first argument is an upper-case string literal. Without them, every knob read
# through a wrapper looked unread and the reverse check below could not tell a
# dead variable from a live one.
_ENV_READ = (
    r"(?:os\.(?:getenv|environ\.get)|\b[A-Za-z_]*env[A-Za-z_]*)"
    r"\(\s*[\"']([A-Z][A-Z0-9_]*)[\"']"
)


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
        names = set(re.findall(_ENV_READ, source))
        for call in re.findall(
            r"ttl_from_env\(\s*os\.(?:getenv|environ\.get)\s*,([^)]*)\)", source
        ):
            names.update(re.findall(r"[\"']([A-Z0-9_]+)[\"']", call))
        out.setdefault(path.parent.name, set()).update(names)
    return out


def _compose_environment() -> dict[str, set[str]]:
    """{service_name: keys in its environment: mapping}."""
    document = load_compose(COMPOSE)
    out: dict[str, set[str]] = {}
    for name, service in (document.get("services") or {}).items():
        out[name] = set(env_mapping(service))
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
    # the SafeLoader knows; load_compose resolves them to their untagged value.
    all_keys: set[str] = set()
    for compose in REPO_ROOT.glob("docker-compose*.yml"):
        document = load_compose(compose)
        for service in (document.get("services") or {}).values():
            all_keys |= set(env_mapping(service or {}))
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
    return set(env_mapping((document.get("services") or {}).get(service, {}) or {}))


def test_truenas_app_matches_the_base_stack():
    """Every env key the base sets must also be set in the standalone file.

    Extra keys there are fine — it pins a GPU the base leaves flexible. Missing
    keys are not: they silently split the fleet.
    """
    base = load_compose(COMPOSE)
    standalone = load_compose(TRUENAS_APP)

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


# --- the same default everywhere ---------------------------------------------
#
# Forwarding a variable is half of it. `MAX_TTS_CHARS` was 20000 in three
# deployment files and 5000 in the code that reads it, for as long as nobody
# looked: every install "used the default" and got a value the documentation
# and the service disagreed about. The rules below make the file, not a person,
# notice.

# Keys whose value is meant to differ between the base stack and the standalone
# TrueNAS file, with the reason. GPU pinning is PARITY_IGNORE above.
DEFAULT_DIFFERS = {
    ("stt-service", "STT_DEFAULT_LANGUAGE"): "the TrueNAS file is German-first; the base file detects",
    ("stt-service", "WHISPER_COMPUTE_TYPE"): "the TrueNAS file targets one NVIDIA card (float16); the base file picks per device",
    ("frontend-service", "ENABLE_TRAINING"): "the TrueNAS file installs no training container until its profiles line goes, "
                                             "so its tab would be a red dot; the base file keeps the tab compose users had",
}

_NUMBER = r"(-?\d+(?:\.\d+)?)"
# A numeric default written next to the name at the read: `env_number("X", 4)`,
# `_number(os.getenv("X"), "X", 60.0, ...)`, `os.getenv("X", "512")`.
_DEFAULT_READ = (
    re.compile(r"\b[A-Za-z_]*env[A-Za-z_]*\(\s*[\"']([A-Z][A-Z0-9_]*)[\"']\s*,\s*" + _NUMBER + r"\s*[,)]"),
    re.compile(
        r"_number\(\s*os\.(?:getenv|environ\.get)\(\s*[\"']([A-Z][A-Z0-9_]*)[\"']\)\s*,\s*"
        r"[\"'][A-Z0-9_]+[\"']\s*,\s*" + _NUMBER + r"\s*[,)]"),
    re.compile(r"os\.(?:getenv|environ\.get)\(\s*[\"']([A-Z][A-Z0-9_]*)[\"']\s*,\s*[\"']" + _NUMBER + r"[\"']\s*\)"),
)


def _code_defaults() -> dict[str, dict[str, set[float]]]:
    """{service_dir: {variable: numeric defaults its Python gives it}}."""
    out: dict[str, dict[str, set[float]]] = {}
    for path in REPO_ROOT.glob("*-service/*.py"):
        source = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in _DEFAULT_READ:
            for name, value in pattern.findall(source):
                out.setdefault(path.parent.name, {}).setdefault(name, set()).add(float(value))
    return out


def _numeric_default(value) -> float | None:
    """The number a compose value resolves to with nothing set, or None (empty, or not a number)."""
    text = interpolate(str(value)).strip()
    return float(text) if re.fullmatch(_NUMBER, text) else None


def test_a_compose_default_is_the_default_the_code_would_use_anyway():
    """`${MAX_TTS_CHARS:-20000}` next to `_env_number("MAX_TTS_CHARS", 5000, int)` is a lie.

    An empty default is fine (it means "the code decides", which is how derived
    defaults such as 4 x ASR_MAX_CONCURRENCY are spelled). A number must be the
    number the code falls back to, in the base file and in the standalone one.
    """
    code = _code_defaults()
    wrong: list[str] = []
    checked = 0
    for compose in (COMPOSE, TRUENAS_APP):
        services = load_compose(compose).get("services") or {}
        for service, definition in sorted(services.items()):
            for key, value in sorted(env_mapping(definition or {}).items()):
                defaults = code.get(service, {}).get(key)
                number = _numeric_default(value)
                if not defaults or number is None:
                    continue
                checked += 1
                if number not in defaults:
                    wrong.append(
                        f"{compose.name}: {service}.{key} defaults to {number:g}, "
                        f"the code to {sorted(defaults)}")
    assert not wrong, "compose disagrees with the service about a default:\n  " + "\n  ".join(wrong)
    # The scan itself must keep finding things, or it guards nothing.
    assert checked >= 40, f"only {checked} defaults were comparable; did the read helpers change shape?"


def test_the_limit_knobs_are_forwarded_with_the_code_default_everywhere():
    """The knobs the hardening added, by name, in every file that deploys the service."""
    expected = {
        "frontend-service": {
            "TRUSTED_HOSTS": "", "ALLOWED_HOSTS": "", "MAX_TTS_CHARS": "5000",
            "MAX_CONCURRENT_UPLOADS": "4", "MAX_CONCURRENT_FFMPEG": "4",
        },
        "piper-tts-service": {
            "PIPER_ANALYZE_MAX_SECONDS": "600", "PIPER_MAX_CUSTOM_VOICES": "20",
            "PIPER_MAX_CUSTOM_MB": "2048", "PIPER_ONNX_VALIDATE_TIMEOUT_S": "30",
        },
        "qwen3-asr-service": {"ASR_MAX_QUEUE": "", "ASR_QUEUE_TIMEOUT_S": "60"},
        "parakeet-asr-service": {"ASR_MAX_QUEUE": "", "ASR_QUEUE_TIMEOUT_S": "60"},
        "canary-asr-service": {"ASR_MAX_QUEUE": "", "ASR_QUEUE_TIMEOUT_S": "60"},
        "qwen3-tts-service": {
            "QWEN3_TTS_REF_MAX_SECONDS": "60", "TTS_MAX_QUEUE": "", "TTS_QUEUE_TIMEOUT_S": "60",
        },
        "chatterbox-tts-service": {"TTS_MAX_QUEUE": "", "TTS_QUEUE_TIMEOUT_S": "60"},
        "magpie-tts-service": {"TTS_MAX_QUEUE": "", "TTS_QUEUE_TIMEOUT_S": "60"},
    }
    problems: list[str] = []
    for compose in (COMPOSE, TRUENAS_APP):
        services = load_compose(compose).get("services") or {}
        for service, knobs in expected.items():
            environment = env_mapping(services.get(service) or {})
            for key, default in knobs.items():
                if key not in environment:
                    problems.append(f"{compose.name}: {service} does not forward {key}")
                elif interpolate(str(environment[key])) != default:
                    problems.append(
                        f"{compose.name}: {service}.{key} defaults to "
                        f"{interpolate(str(environment[key]))!r}, expected {default!r}")
    assert not problems, "\n  ".join(problems)


def test_canary_is_not_handed_a_batch_size_it_does_not_read():
    """Canary decodes one file per pass; ASR_MAX_BATCH belongs to Parakeet alone."""
    for compose in sorted(REPO_ROOT.glob("docker-compose*.yml")):
        services = load_compose(compose).get("services") or {}
        assert "ASR_MAX_BATCH" not in env_mapping(services.get("canary-asr-service") or {}), compose.name
    template = (REPO_ROOT / "truenas/tts-stt/templates/docker-compose.yaml").read_text(encoding="utf-8")
    block = template.split("canary-asr-service:", 1)[1].split("\n\n", 1)[0]
    assert "ASR_MAX_BATCH" not in block


def test_truenas_app_defaults_match_the_base_stack():
    """The same key must resolve to the same default in both files, unless the difference is listed."""
    base = load_compose(COMPOSE).get("services") or {}
    standalone = load_compose(TRUENAS_APP).get("services") or {}
    differing: list[str] = []
    for service in sorted(standalone):
        if service not in base:
            continue
        a, b = env_mapping(base[service]), env_mapping(standalone[service])
        for key in sorted((set(a) & set(b)) - PARITY_IGNORE):
            if (service, key) in DEFAULT_DIFFERS:
                continue
            if interpolate(str(a[key])) != interpolate(str(b[key])):
                differing.append(
                    f"{service}.{key}: base {interpolate(str(a[key]))!r}, "
                    f"standalone {interpolate(str(b[key]))!r}")
    assert not differing, (
        "docker-compose.truenas-app.yml and docker-compose.yml resolve the same variable to different "
        "defaults, so a TrueNAS install and a compose install of one commit disagree:\n  "
        + "\n  ".join(differing)
        + "\nFix the file that is wrong, or list the pair in DEFAULT_DIFFERS with the reason."
    )


def test_every_default_differs_entry_still_differs():
    """An entry for a pair that agrees again would hide the next real drift of that pair."""
    base = load_compose(COMPOSE).get("services") or {}
    standalone = load_compose(TRUENAS_APP).get("services") or {}
    stale = []
    for service, key in DEFAULT_DIFFERS:
        a = env_mapping(base.get(service) or {}).get(key)
        b = env_mapping(standalone.get(service) or {}).get(key)
        if a is None or b is None or interpolate(str(a)) == interpolate(str(b)):
            stale.append(f"{service}.{key}")
    assert not stale, f"DEFAULT_DIFFERS lists pairs that no longer differ: {stale}"


# --- the Settings page's own switches -------------------------------------------
#
# Three gateway variables came with the Settings page (web UI -> gear). Two are the
# switches the page must never hold over itself, so they exist only in the deployment
# files: ENABLE_SETTINGS_UI=false makes the page read-only and its saved values
# ignored, SETTINGS_LOCKED_KEYS pins settings to the file. The third, ENABLE_TRAINING,
# hides voice training; the TrueNAS file defaults it to false because it installs no
# training container (see DEFAULT_DIFFERS). Each must reach the gateway in every file
# that deploys it, with the same default as .env.example says.

SETTINGS_SWITCHES = {
    # name: (default in docker-compose.yml and .env.example, default in the TrueNAS file)
    "ENABLE_SETTINGS_UI": ("true", "true"),
    "SETTINGS_LOCKED_KEYS": ("", ""),
    "ENABLE_TRAINING": ("true", "false"),
}


def test_the_settings_page_switches_reach_the_gateway_everywhere_with_their_defaults():
    documented = {name: value.strip() for name, value in
                  re.findall(r"^([A-Z0-9_]+)=(.*)$", ENV_EXAMPLE.read_text(encoding="utf-8"), flags=re.M)}
    base = env_mapping(load_compose(COMPOSE)["services"]["frontend-service"])
    standalone = env_mapping(load_compose(TRUENAS_APP)["services"]["frontend-service"])
    template = (REPO_ROOT / "truenas/tts-stt/templates/docker-compose.yaml").read_text(encoding="utf-8")
    frontend_block = template.split("\n  frontend-service:", 1)[1].split("\n\n", 1)[0]
    read = _consumed_by_service()["frontend-service"]

    problems = []
    for name, (base_default, truenas_default) in SETTINGS_SWITCHES.items():
        if name not in read:
            problems.append(f"frontend-service never reads {name}")
        for label, environment, default in (("docker-compose.yml", base, base_default),
                                             ("docker-compose.truenas-app.yml", standalone, truenas_default)):
            if name not in environment:
                problems.append(f"{label}: frontend-service does not get {name}")
            elif interpolate(str(environment[name])) != default:
                problems.append(f"{label}: {name} defaults to {interpolate(str(environment[name]))!r}, "
                                f"expected {default!r}")
        if documented.get(name) != base_default:
            problems.append(f".env.example: {name}={documented.get(name)!r}, expected {base_default!r}")
        if f"\n      {name}: " not in frontend_block:
            problems.append(f"truenas template: frontend-service does not set {name}")
    assert not problems, "\n  ".join(problems)


def test_the_settings_folders_are_documented_as_path_overrides():
    """SETTINGS_DIR and BACKEND_SETTINGS_DIR move the two folders, like MODELS_DIR moves models/.

    They are mounts, not environment (the _DIR rule above), and are listed commented out in
    .env.example with the other path overrides, since their default follows APP_DATA_DIR.
    """
    example = ENV_EXAMPLE.read_text(encoding="utf-8")
    compose = COMPOSE.read_text(encoding="utf-8")
    for name, folder in (("SETTINGS_DIR", "settings"), ("BACKEND_SETTINGS_DIR", "backend-settings")):
        assert re.search(rf"^#\s*{name}=\s*$", example, flags=re.M), f".env.example does not document {name}"
        assert f"${{{name}:-${{APP_DATA_DIR:-.}}/{folder}}}" in compose, f"docker-compose.yml does not read {name}"


# --- the documentation has to say what the services do -----------------------

# The limits the hardening added. Each must be in .env.example (test above), in
# docs/api.md's limits tables and in the TrueNAS Custom App file.
DOCUMENTED_KNOBS = (
    "TRUSTED_HOSTS", "ALLOWED_HOSTS", "MAX_CONCURRENT_UPLOADS", "MAX_CONCURRENT_FFMPEG",
    "ASR_MAX_QUEUE", "ASR_QUEUE_TIMEOUT_S", "TTS_MAX_QUEUE", "TTS_QUEUE_TIMEOUT_S",
    "PIPER_ANALYZE_MAX_SECONDS", "PIPER_MAX_CUSTOM_VOICES", "PIPER_MAX_CUSTOM_MB",
    "PIPER_ONNX_VALIDATE_TIMEOUT_S", "QWEN3_TTS_REF_MAX_SECONDS",
)


def test_every_limit_knob_is_in_the_env_file_the_api_docs_and_the_truenas_form_files():
    api = (REPO_ROOT / "docs" / "api.md").read_text(encoding="utf-8")
    truenas_app = TRUENAS_APP.read_text(encoding="utf-8")
    template = (REPO_ROOT / "truenas/tts-stt/templates/docker-compose.yaml").read_text(encoding="utf-8")
    documented = _documented()
    missing = []
    for knob in DOCUMENTED_KNOBS:
        if knob not in documented:
            missing.append(f".env.example: {knob}")
        if knob not in api:
            missing.append(f"docs/api.md: {knob}")
        if knob not in truenas_app:
            missing.append(f"docker-compose.truenas-app.yml: {knob}")
        if knob not in template:
            missing.append(f"truenas template: {knob}")
    assert not missing, "a limit knob is not written down everywhere it has to be:\n  " + "\n  ".join(missing)


def test_a_hostname_that_needs_trusted_hosts_is_asked_for_in_the_truenas_form():
    """A reverse proxy or a real domain is 403 host_not_allowed without it; an operator must be able to set it."""
    import yaml

    form = yaml.safe_load((REPO_ROOT / "truenas/tts-stt/questions.yaml").read_text(encoding="utf-8"))
    names = {question["variable"] for question in form["questions"]}
    assert "trusted_hosts" in names
    guide = (REPO_ROOT / "docs" / "truenas-installation-guide.md").read_text(encoding="utf-8")
    assert "host_not_allowed" in guide and "TRUSTED_HOSTS" in guide


# Statements that were true before the gateway validated Host, stopped exempting
# its own UI from API_KEY, resampled pcm to 24 kHz and lowered MAX_TTS_CHARS.
STALE_STATEMENTS = (
    "same-origin browser calls are exempt",
    "same-origin browser requests do not",
    "except same-origin browser requests",
    "except requests the bundled UI itself makes",
    "same-origin browser request). Reads",
    "at the backend's **native",
    "It is not resampled here",
    "it does not guard `/ws/stt`",
    "and `API_KEY` does not",
    "`512`, `20000`",
    "MAX_TTS_CHARS:-20000",
    "MAX_TTS_CHARS=20000",
    "(default 20000, 422 beyond it)",
)


def test_no_document_or_deployment_file_repeats_a_retired_statement():
    paths = [
        *REPO_ROOT.glob("docker-compose*.yml"), *REPO_ROOT.glob(".env*example"),
        REPO_ROOT / "README.md", *(REPO_ROOT / "docs").glob("*.md"),
        *(REPO_ROOT / "truenas").rglob("*.md"), *(REPO_ROOT / "truenas").rglob("*.yaml"),
    ]
    found = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for statement in STALE_STATEMENTS:
            if statement in text:
                found.append(f"{path.relative_to(REPO_ROOT)}: {statement!r}")
    assert not found, "these statements are no longer true:\n  " + "\n  ".join(found)
