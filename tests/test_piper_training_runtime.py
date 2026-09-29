"""How the training service is started: keep-alive against the gateway's pool.

The gateway (frontend-service) pools upstream connections and keeps an idle one for
`UPSTREAM_KEEPALIVE_EXPIRY` (115 s). A server that closes idle connections sooner
hands the pool sockets that are already dead, and the request that draws one fails
with a bare "server disconnected". The training service closed at 75 s; the other
backends use 120 s.

The image starts through start.sh, which runs `python3 app.py`, so a
`--timeout-keep-alive` flag on the Dockerfile's CMD (the way the other images set it)
would reach nothing: the value has to be passed to `uvicorn.run` in `app.py`, and
compose passes `UVICORN_TIMEOUT_KEEP_ALIVE` over the default, so its default matters as
much as the code's.
"""

from __future__ import annotations

import re
import runpy

import pytest
from compose_helpers import REPO_ROOT, dotenv, env_mapping, interpolate, load_compose

from test_piper_training_service_support import SERVICE_DIR, training_service  # noqa: F401  (fixture)

GATEWAY_EXPIRY_DEFAULT = 115


def _gateway_expiry_default() -> int:
    """The gateway's default idle time, read from its compose block rather than repeated here."""
    gateway = env_mapping(load_compose(REPO_ROOT / "docker-compose.yml")["services"]["frontend-service"])
    return int(interpolate(str(gateway["UPSTREAM_KEEPALIVE_EXPIRY"])))


def _run_main(training_service, monkeypatch, env=None):
    """Execute app.py as `python3 app.py` would, with uvicorn.run recorded instead of serving."""
    training_service()          # installs the stand-ins app.py imports (torch, librosa, ...)
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)
    import uvicorn
    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: calls.append((args, kwargs)))

    runpy.run_path(str(SERVICE_DIR / "app.py"), run_name="__main__")

    ((args, kwargs),) = calls
    return args, kwargs


def test_the_gateways_expiry_default_is_what_this_file_assumes():
    assert _gateway_expiry_default() == GATEWAY_EXPIRY_DEFAULT


def test_running_the_module_directly_keeps_idle_connections_past_the_gateways_pool(training_service, monkeypatch):
    _, kwargs = _run_main(training_service, monkeypatch)

    assert kwargs["port"] == 8080
    assert kwargs["timeout_keep_alive"] == 120
    assert kwargs["timeout_keep_alive"] > _gateway_expiry_default()


def test_the_environment_can_still_override_the_keep_alive(training_service, monkeypatch):
    _, kwargs = _run_main(training_service, monkeypatch, {"UVICORN_TIMEOUT_KEEP_ALIVE": "150"})
    assert kwargs["timeout_keep_alive"] == 150


@pytest.mark.parametrize("junk", ["soon", "0", "-5", "nan"])
def test_an_unusable_override_falls_back_to_the_default_instead_of_stopping_the_service(
        training_service, monkeypatch, junk):
    _, kwargs = _run_main(training_service, monkeypatch, {"UVICORN_TIMEOUT_KEEP_ALIVE": junk})
    assert kwargs["timeout_keep_alive"] == 120


def test_compose_no_longer_overrides_the_code_default_with_a_shorter_one():
    """compose passes UVICORN_TIMEOUT_KEEP_ALIVE, so its default is what a deployment runs."""
    service = load_compose(REPO_ROOT / "docker-compose.yml")["services"]["piper-training-service"]
    value = int(interpolate(str(env_mapping(service)["UVICORN_TIMEOUT_KEEP_ALIVE"])))

    assert value >= _gateway_expiry_default(), (
        f"the training service closes idle connections after {value} s, before the gateway's "
        f"{_gateway_expiry_default()} s pool expiry: pooled sockets come back dead")
    assert value == 120


def test_the_example_env_file_and_the_readme_document_the_same_default():
    example = int(dotenv(REPO_ROOT / ".env.example")["UVICORN_TIMEOUT_KEEP_ALIVE"])
    readme = (SERVICE_DIR / "README.md").read_text(encoding="utf-8")
    row = re.search(r"\|\s*`UVICORN_TIMEOUT_KEEP_ALIVE`\s*\|\s*`(\d+)`", readme)

    assert example == 120
    assert row and int(row.group(1)) == 120


def test_the_dockerfile_cmd_is_start_sh_so_the_flag_lives_in_app_py():
    """If this ever becomes a uvicorn command line, the flag belongs there instead."""
    for name in ("Dockerfile", "Dockerfile.rocm"):
        text = (SERVICE_DIR / name).read_text(encoding="utf-8")
        cmd = re.findall(r"^CMD\s+(.*)$", text, flags=re.M)
        assert cmd == ['["/app/start.sh"]'], (name, cmd)
    assert "exec python3 app.py" in (SERVICE_DIR / "start.sh").read_text(encoding="utf-8")
