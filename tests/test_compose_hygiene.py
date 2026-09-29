"""Guards for what the base compose file promises about ports, health and updates.

These read the compose files the way Compose would (variable substitution
included) and assert on the resulting deployment, not on how the YAML is
spelled. Each one exists because the opposite state was shipped:

* every backend port was published on 0.0.0.0 while the docs said publishing was
  unnecessary, and the backends have no authentication;
* the ROCm overlay merged the CUDA build args into the ROCm image, which would
  have silently built it against NeMo 3 on a torch below NeMo's floor;
* the training service got Docker's default 10 s to stop, so every image update
  killed a running job.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from compose_helpers import (
    REPO_ROOT,
    Tagged,
    dotenv,
    env_mapping,
    interpolate,
    load_compose,
    load_compose_with_tags,
)

BASE = REPO_ROOT / "docker-compose.yml"
GATEWAY = "frontend-service"


def _services() -> dict:
    return load_compose(BASE)["services"]


def _ports(service: dict, env: dict | None = None) -> list[tuple[str, int, int]]:
    """[(host_ip, published, target)] after substitution; no host_ip means all interfaces."""
    out = []
    for entry in service.get("ports") or []:
        parts = interpolate(str(entry), env).split(":")
        if len(parts) == 3:
            out.append((parts[0], int(parts[1]), int(parts[2])))
        else:
            out.append(("0.0.0.0", int(parts[0]), int(parts[1])))
    return out


def _seconds(duration: str) -> float:
    """'60s' / '1m30s' / '2h' -> seconds (the forms Compose accepts)."""
    total = 0.0
    for amount, unit in re.findall(r"(\d+(?:\.\d+)?)(ms|us|s|m|h)", str(duration)):
        total += float(amount) * {"us": 1e-6, "ms": 1e-3, "s": 1, "m": 60, "h": 3600}[unit]
    if not total:
        raise AssertionError(f"cannot read the duration {duration!r}")
    return total


# --- ports -----------------------------------------------------------------------


def test_only_the_gateway_is_published_beyond_loopback_by_default():
    """The frontend proxies everything; the unauthenticated backends stay on 127.0.0.1."""
    exposed = {}
    for name, service in _services().items():
        for host_ip, published, _target in _ports(service):
            if host_ip not in ("127.0.0.1", "::1", "localhost"):
                exposed[name] = (host_ip, published)

    assert exposed == {GATEWAY: ("0.0.0.0", 3000)}, (
        "these services publish a port on a non-loopback address by default, and "
        f"only {GATEWAY} may: {exposed}. Bind backends with "
        "${BACKEND_BIND_ADDR:-127.0.0.1}."
    )


def test_every_backend_still_publishes_its_port_for_local_tools():
    """Loopback-only is not 'unpublished': curl, benchmarks and scripts use these."""
    missing = [
        name for name, service in _services().items()
        if name != GATEWAY and service.get("profiles") and name != "piper-voices-seed"
        and not _ports(service)
    ]
    assert not missing, f"backends without a published port: {missing}"


def test_the_bind_address_variables_change_the_published_address():
    """Behavioural: BACKEND_BIND_ADDR moves the backends, FRONTEND_BIND_ADDR the gateway."""
    services = _services()
    opened = {"BACKEND_BIND_ADDR": "0.0.0.0", "FRONTEND_BIND_ADDR": "127.0.0.1"}
    for name, service in services.items():
        published = _ports(service, opened)
        if not published:
            continue
        expected = "127.0.0.1" if name == GATEWAY else "0.0.0.0"
        assert {ip for ip, _, _ in published} == {expected}, (
            f"{name} ignores {'FRONTEND' if name == GATEWAY else 'BACKEND'}_BIND_ADDR: {published}"
        )


def test_the_bind_addresses_are_documented_with_the_safe_defaults():
    documented = dotenv(REPO_ROOT / ".env.example")
    assert documented.get("BACKEND_BIND_ADDR") == "127.0.0.1"
    assert documented.get("FRONTEND_BIND_ADDR") == "0.0.0.0"


_URL = re.compile(r"^https?://", re.I)


@pytest.mark.parametrize("compose_name", ["docker-compose.yml", "docker-compose.truenas-app.yml"])
def test_no_service_reaches_another_through_a_published_port(compose_name):
    """Services talk over the compose network by service name and container port.

    A URL that points at localhost, the Docker host, or a published (host) port
    works on the developer's machine and breaks the moment the port binding, the
    host port or the network changes -- and would defeat BACKEND_BIND_ADDR.
    """
    path = REPO_ROOT / compose_name
    if not path.exists():
        pytest.skip(f"{compose_name} not present")
    services = load_compose(path)["services"]
    container_ports = {
        name: {target for _, _, target in _ports(svc)} for name, svc in services.items()
    }
    # the standalone TrueNAS file keeps backend ports as comments; use the
    # container ports named by the base file for names it does not publish itself
    base_ports = {
        name: {target for _, _, target in _ports(svc)} for name, svc in _services().items()
    }

    problems = []
    for name, service in services.items():
        for key, raw in env_mapping(service).items():
            value = interpolate(str(raw))
            if not _URL.match(value):
                continue
            parts = urlsplit(value)
            host, port = parts.hostname, parts.port
            if host in ("localhost", "127.0.0.1", "0.0.0.0", "host.docker.internal") or host is None:
                problems.append(f"{name}: {key}={value} points at the host, not a service")
                continue
            if "." in host:
                continue  # an external endpoint (a notification service, a registry), not a peer
            if host not in services and host not in base_ports:
                problems.append(f"{name}: {key}={value}: no service called {host!r}")
                continue
            known = container_ports.get(host) or base_ports.get(host) or set()
            if port is not None and known and port not in known:
                problems.append(
                    f"{name}: {key}={value} uses port {port}, but {host} listens on "
                    f"{sorted(known)} (a published host port is not reachable that way)"
                )
    assert not problems, "\n".join(problems)


# --- health and startup order ---------------------------------------------------------


def test_healthchecks_are_liveness_probes():
    """/health (or, for whisper-server, any HTTP answer) -- never /ready.

    A model download takes minutes and optional backends may be absent, so a
    readiness gate would keep the container "unhealthy" and everything that
    waits on it down.
    """
    bad = []
    for name, service in _services().items():
        check = service.get("healthcheck")
        if not check:
            continue
        command = " ".join(check["test"][1:]) if isinstance(check["test"], list) else str(check["test"])
        if "/ready" in command:
            bad.append(f"{name}: {command}")
    assert not bad, f"healthchecks must not gate on /ready: {bad}"


def test_the_gateway_never_waits_for_a_backend_to_be_ready_or_healthy():
    """service_started only: optional backends and slow model loads must not hold the UI back."""
    depends = _services()[GATEWAY].get("depends_on") or {}
    conditions = {dep: spec.get("condition") for dep, spec in depends.items()}
    assert conditions, "the gateway is expected to start after piper and stt where present"
    assert set(conditions.values()) == {"service_started"}, conditions
    assert all(spec.get("required") is False for spec in depends.values()), (
        "long form with required: false, or a profile selection that omits a "
        "dependency makes the whole project invalid"
    )


# --- updates ---------------------------------------------------------------------------


def test_training_gets_longer_to_stop_than_it_waits_for_its_jobs():
    """SIGTERM -> jobs stop at an epoch boundary. Docker must not SIGKILL before that."""
    service = _services()["piper-training-service"]
    env = env_mapping(service)
    wait = float(interpolate(str(env["TRAINING_SHUTDOWN_GRACE_S"])))
    grace = _seconds(service["stop_grace_period"])
    assert grace > wait, (
        f"stop_grace_period {grace:.0f}s is not longer than TRAINING_SHUTDOWN_GRACE_S "
        f"{wait:.0f}s: Docker would kill the container while it is still saving job state"
    )


def test_the_documented_shutdown_wait_fits_inside_the_grace_period():
    """The documented default and the compose default agree (see the wiring suite)."""
    documented = dotenv(REPO_ROOT / ".env.example")
    assert float(documented["TRAINING_SHUTDOWN_GRACE_S"]) < _seconds(
        _services()["piper-training-service"]["stop_grace_period"])


MODEL_SERVICES = (
    "stt-service", "qwen3-asr-service", "qwen3-tts-service",
    "parakeet-asr-service", "canary-asr-service", "chatterbox-tts-service",
)


@pytest.mark.parametrize("name", MODEL_SERVICES)
def test_model_services_get_more_than_dockers_default_to_stop(name):
    assert _seconds(_services()[name].get("stop_grace_period", "10s")) > 10, (
        f"{name} has the 10 s default: a decode in flight or a live session is "
        "killed by every `docker compose up` that updates it"
    )


def test_every_image_service_lets_the_operator_choose_the_pull_policy():
    """PULL_POLICY=always is how an update to a moved tag (latest) is fetched."""
    missing = []
    for name, service in _services().items():
        if "image" not in service:
            continue
        policy = service.get("pull_policy")
        if policy != "${PULL_POLICY:-missing}":
            missing.append(f"{name}: {policy!r}")
    assert not missing, f"pull_policy must be ${{PULL_POLICY:-missing}}: {missing}"
    assert dotenv(REPO_ROOT / ".env.example").get("PULL_POLICY") == "missing"


def test_resource_limits_default_to_unlimited():
    """Optional caps must add no limit unless the operator sets one."""
    service = _services()["piper-training-service"]
    for key in ("mem_limit", "cpus"):
        assert key in service, f"{key} is expected as an opt-in knob"
        assert str(interpolate(str(service[key]))) in ("0", "0.0"), (
            f"{key} defaults to {interpolate(str(service[key]))!r}, which is a limit"
        )
    limited = interpolate(str(service["mem_limit"]), {"TRAINING_MEM_LIMIT": "12g"})
    assert limited == "12g", "the knob does not reach the service"


# --- build ---------------------------------------------------------------------------------


def _dockerfile_args(context: str, dockerfile: str) -> set[str]:
    text = (REPO_ROOT / context / dockerfile).read_text(encoding="utf-8").replace("\\\n", " ")
    return set(re.findall(r"^\s*ARG\s+([A-Za-z_][A-Za-z0-9_]*)", text, flags=re.M))


def _build_arg_names(args) -> list[str]:
    if isinstance(args, dict):
        return list(args)
    return [str(entry).split("=", 1)[0] for entry in args or []]


def test_build_args_are_declared_by_the_dockerfile_that_receives_them():
    """An arg the Dockerfile never declares is silently ignored: the pin does nothing."""
    problems = []
    for name, service in _services().items():
        build = service.get("build")
        if not isinstance(build, dict):
            continue
        declared = _dockerfile_args(
            build["context"].lstrip("./"), build.get("dockerfile", "Dockerfile"))
        for arg in _build_arg_names(build.get("args")):
            if arg not in declared:
                problems.append(f"{name}: build arg {arg} is not an ARG in "
                                f"{build['context']}/{build.get('dockerfile', 'Dockerfile')}")
    assert not problems, "\n".join(problems)


def test_pins_stay_in_the_dockerfile_not_in_compose():
    """Version pins have one home. Compose may only override them, never restate them
    with a different default than the Dockerfile, or a local build and the
    published image (which ignores compose) would differ."""
    problems = []
    for name, service in _services().items():
        build = service.get("build")
        if not isinstance(build, dict) or not build.get("args"):
            continue
        context, dockerfile = build["context"].lstrip("./"), build.get("dockerfile", "Dockerfile")
        text = (REPO_ROOT / context / dockerfile).read_text(encoding="utf-8").replace("\\\n", " ")
        defaults = dict(re.findall(r'^\s*ARG\s+([A-Za-z_]\w*)=["\']?([^"\'\s]+)', text, flags=re.M))
        for entry in build["args"]:
            key, _, value = str(entry).partition("=")
            if not value:
                continue
            resolved = interpolate(value)
            if key in defaults and resolved != defaults[key]:
                problems.append(
                    f"{name}: compose defaults {key} to {resolved!r} but the Dockerfile "
                    f"says {defaults[key]!r}")
    assert not problems, "\n".join(problems)


def test_the_rocm_overlay_does_not_inherit_the_cuda_build_args():
    """A merge would give the ROCm parakeet image NeMo 3 on torch 2.5.1 (below the floor)."""
    base = _services()
    overlay = load_compose_with_tags(REPO_ROOT / "docker-compose.rocm.yml")["services"]
    leaking = []
    for name, service in overlay.items():
        if not (base.get(name, {}).get("build") or {}).get("args"):
            continue
        args = (service.get("build") or {}).get("args")
        if not (isinstance(args, Tagged) and args.tag in {"reset", "override"}):
            leaking.append(name)
    assert not leaking, (
        f"{leaking} inherit build args from the CUDA base file. Reset them with "
        "`args: !reset null` or replace them with `!override`."
    )


# --- the voice seed ------------------------------------------------------------------------


def _seed_script() -> str:
    seed = _services()["piper-voices-seed"]
    assert seed["entrypoint"] == ["sh", "-c"]
    return interpolate(seed["command"][0])


def _run_seed(image_dir: Path, models_dir: Path) -> subprocess.CompletedProcess:
    script = (_seed_script()
              .replace("/app/models/default", str(image_dir))
              .replace("/seed/default", str(models_dir / "default")))
    return subprocess.run(["sh", "-c", script], capture_output=True, text=True, timeout=30)


def test_the_seed_copies_missing_voices_and_never_overwrites_or_repeats(tmp_path):
    """The models directory is a bind mount that hides the voices baked into the image."""
    image_dir, models = tmp_path / "image", tmp_path / "models"
    image_dir.mkdir()
    (image_dir / "de_DE-thorsten-medium.onnx").write_text("image onnx")
    (image_dir / "de_DE-thorsten-medium.onnx.json").write_text("{}")

    first = _run_seed(image_dir, models)
    assert first.returncode == 0, first.stderr
    assert (models / "default" / "de_DE-thorsten-medium.onnx").read_text() == "image onnx"
    assert "2 new voice file(s)" in first.stdout

    # an operator's own copy of a voice must survive, and nothing is copied twice
    (models / "default" / "de_DE-thorsten-medium.onnx").write_text("operator edit")
    second = _run_seed(image_dir, models)
    assert second.returncode == 0, second.stderr
    assert (models / "default" / "de_DE-thorsten-medium.onnx").read_text() == "operator edit"
    assert "0 new voice file(s)" in second.stdout

    # a voice added to the image later arrives on the next start
    (image_dir / "en_US-amy-medium.onnx").write_text("amy")
    third = _run_seed(image_dir, models)
    assert (models / "default" / "en_US-amy-medium.onnx").read_text() == "amy"
    assert "1 new voice file(s)" in third.stdout
    assert not list((models / "default").glob("*.part")), "a half-copied file was left behind"


def test_the_seed_fails_loudly_when_the_models_directory_is_not_writable(tmp_path):
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")
    image_dir, models = tmp_path / "image", tmp_path / "models"
    image_dir.mkdir()
    (image_dir / "v.onnx").write_text("x")
    models.mkdir()
    models.chmod(0o500)
    try:
        result = _run_seed(image_dir, models)
    finally:
        models.chmod(0o700)
    assert result.returncode != 0, "a seed that cannot write must not report success"


def test_piper_starts_only_after_the_seed_finished_and_the_seed_uses_the_same_image():
    services = _services()
    piper, seed = services["piper-tts-service"], services["piper-voices-seed"]
    assert piper["depends_on"]["piper-voices-seed"]["condition"] == "service_completed_successfully"
    assert seed["image"] == piper["image"], "the seed must copy the voices of the image Piper runs"
    assert seed["restart"] == "no"
    assert set(seed["profiles"]) == set(piper["profiles"]), (
        "profiles differ: selecting Piper alone would leave the seed out or in")
    seed_mounts = [interpolate(str(v)) for v in seed["volumes"]]
    piper_models = [interpolate(str(v)) for v in piper["volumes"] if str(v).endswith(":/app/models")]
    assert [m.rsplit(":", 1)[0] for m in seed_mounts] == [m.rsplit(":", 1)[0] for m in piper_models], (
        "the seed writes to a different host directory than the one Piper reads")


# --- the compose CLI --------------------------------------------------------------------------

_OVERLAYS = [
    ("docker-compose.yml",),
    ("docker-compose.yml", "docker-compose.rocm.yml"),
    ("docker-compose.yml", "docker-compose.vulkan.yml"),
    ("docker-compose.yml", "docker-compose.arm64.yml"),
    ("docker-compose.yml", "docker-compose.dev.yml"),
    ("docker-compose.yml", "docker-compose.truenas.yml"),
    ("docker-compose.yml", "tests/docker-compose.test.yml"),
]


def _docker_compose() -> list[str] | None:
    if not shutil.which("docker"):
        return None
    probe = subprocess.run(["docker", "compose", "version"], capture_output=True, text=True)
    return ["docker", "compose"] if probe.returncode == 0 else None


def _need_docker():
    cli = _docker_compose()
    if cli is None:
        if os.environ.get("REQUIRE_DRIFT_SUITES") == "1":
            pytest.fail("the docker compose CLI is required in CI to validate the overlays")
        pytest.skip("docker compose CLI not installed")
    return cli


@pytest.mark.parametrize("files", _OVERLAYS, ids=lambda f: "+".join(Path(x).stem for x in f))
def test_every_overlay_combination_is_valid_compose(files):
    """`docker compose config` needs no daemon; it is the parser deployments will use."""
    cli = _need_docker()
    command = [*cli, "--env-file", ".env.example", "--profile", "all", "--profile", "training"]
    for file in files:
        command += ["-f", file]
    result = subprocess.run([*command, "config", "-q"], cwd=REPO_ROOT, capture_output=True,
                            text=True, timeout=120)
    assert result.returncode == 0, result.stderr


def test_compose_renders_the_bind_addresses_it_promises():
    """The real parser agrees with the interpolation the tests above rely on."""
    cli = _need_docker()
    import json
    rendered = subprocess.run(
        [*cli, "--profile", "all", "--profile", "training", "-f", "docker-compose.yml",
         "config", "--format", "json"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
        env={**os.environ, "BACKEND_BIND_ADDR": "", "FRONTEND_BIND_ADDR": ""},
    )
    assert rendered.returncode == 0, rendered.stderr
    services = json.loads(rendered.stdout)["services"]
    hosts = {
        name: {p.get("host_ip") for p in svc.get("ports", [])}
        for name, svc in services.items() if svc.get("ports")
    }
    assert hosts.pop(GATEWAY) == {"0.0.0.0"}
    assert all(ips == {"127.0.0.1"} for ips in hosts.values()), hosts
