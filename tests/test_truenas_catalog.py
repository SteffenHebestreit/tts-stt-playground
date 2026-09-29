"""The TrueNAS catalog scaffold (truenas/tts-stt): the form, the template, and the Custom App file.

Two files describe the same stack: docker-compose.truenas-app.yml (the supported install path) and
the catalog template. The form (questions.yaml) feeds the template. Nothing keeps three hand-written
descriptions of one stack in step except a test, so this one renders the template the way the Apps
UI would (scripts/truenas/render_catalog.py: defaults from the form, hidden answers left out,
unknown names rejected) and compares the result with the compose file service by service.

What it cannot show: that TrueNAS accepts this directory as a catalog. It has never been loaded by
one (truenas/README.md says so).
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from compose_helpers import REPO_ROOT, env_mapping, interpolate, load_compose, require_yaml

yaml = require_yaml()
jinja2 = pytest.importorskip("jinja2", reason="Jinja2 renders the catalog template (tests/requirements.txt)")
from jinja2 import nodes  # noqa: E402

CATALOG = REPO_ROOT / "truenas" / "tts-stt"
COMPOSE_APP = REPO_ROOT / "docker-compose.truenas-app.yml"
DATA = "/mnt/tank/apps/tts-stt"
DOCKER = shutil.which("docker")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rc = _load(REPO_ROOT / "scripts" / "truenas" / "render_catalog.py", "truenas_render_catalog_for_catalog_tests")

OPTIONAL_ON = {"enable_canary": "true", "enable_parakeet": "true", "enable_chatterbox": "true",
               "enable_training": "true", "enable_whisper_cpp": "true"}

needs_docker = pytest.mark.skipif(DOCKER is None, reason="docker CLI not found: `docker compose config` cannot run")


def questions() -> list[dict]:
    return rc.load_questions()


def answers(**overrides) -> dict:
    merged = {"host_path": DATA}
    merged.update(overrides)
    return rc.default_answers(questions(), merged)


def render(**overrides) -> dict:
    """Rendered template, parsed."""
    return yaml.safe_load(rc.render(answers(**overrides), questions=questions()))


def compose_config(text: str) -> dict:
    """`docker compose config` of a rendered file, as JSON (also proves that Compose accepts it)."""
    result = subprocess.run([DOCKER, "compose", "-f", "-", "config", "--format", "json"], input=text,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


# --- the form -----------------------------------------------------------------------------------------


def test_the_form_parses_and_every_question_is_complete():
    document = yaml.safe_load((CATALOG / "questions.yaml").read_text(encoding="utf-8"))
    groups = {g["name"] for g in document["groups"]}
    names = [q["variable"] for q in document["questions"]]
    assert len(names) == len(set(names)), "a variable asked twice"
    for q in document["questions"]:
        where = q["variable"]
        assert q.get("label"), f"{where}: no label"
        assert q["group"] in groups, f"{where}: group {q['group']!r} is not declared"
        schema = q["schema"]
        assert schema["type"] in {"string", "int", "boolean", "hostpath"}, f"{where}: type {schema['type']}"
        for name, operator, _ in schema.get("show_if", []):
            assert name in names and name != where, f"{where}: show_if refers to {name!r}"
            assert operator in {"=", "!="}, f"{where}: show_if operator {operator!r} is not one the renderer knows"


def test_defaults_match_their_declared_type_and_choices():
    for q in questions():
        schema, name = q["schema"], q["variable"]
        if "default" not in schema:
            assert schema.get("required"), f"{name}: no default and not required"
            continue
        default = schema["default"]
        if schema["type"] == "boolean":
            assert isinstance(default, bool), f"{name}: default {default!r} is not a boolean"
        elif schema["type"] == "int":
            assert isinstance(default, int) and not isinstance(default, bool), name
            assert schema.get("min", default) <= default <= schema.get("max", default), f"{name}: default out of range"
        elif schema["type"] == "string":
            assert isinstance(default, str), f"{name}: default {default!r} must be a string (quote a number)"
        if "enum" in schema:
            assert default in {e["value"] for e in schema["enum"]}, f"{name}: default {default!r} not among the choices"


def test_the_secret_is_marked_private_and_the_host_path_is_required_without_a_default():
    by_name = {q["variable"]: q["schema"] for q in questions()}
    assert by_name["api_key"].get("private") is True
    assert by_name["host_path"]["type"] == "hostpath" and by_name["host_path"].get("required")
    assert "default" not in by_name["host_path"], "a guessed data path would install into the wrong place"


def test_there_is_no_browser_url_setting_anywhere():
    """BROWSER_*_URL was documented as live configuration and read by nothing; it is gone."""
    assert not [q["variable"] for q in questions() if "browser" in q["variable"].lower()]
    for scenario in ({}, OPTIONAL_ON):
        for service in render(**scenario)["services"].values():
            assert not [k for k in env_mapping(service) if k.startswith("BROWSER_")]


# --- form and template agree -----------------------------------------------------------------------------


def template_values() -> set[str]:
    """Every `values.<name>` the template reads (parsed, so every branch counts, not only rendered ones)."""
    tree = jinja2.Environment().parse((CATALOG / "templates" / "docker-compose.yaml").read_text(encoding="utf-8"))
    used = set()
    for node in tree.find_all(nodes.Getattr):
        if isinstance(node.node, nodes.Name) and node.node.name == "values":
            used.add(node.attr)
    for node in tree.find_all(nodes.Getitem):
        if isinstance(node.node, nodes.Name) and node.node.name == "values" and isinstance(node.arg, nodes.Const):
            used.add(node.arg.value)
    return used


# TrueNAS injects this one; it is not a question.
INJECTED = {"ix_volumes"}


def test_every_question_is_used_by_the_template():
    unused = {q["variable"] for q in questions()} - template_values()
    assert not unused, f"asked but never used, so answering it does nothing: {sorted(unused)}"


def test_every_value_the_template_reads_is_asked():
    unasked = template_values() - {q["variable"] for q in questions()} - INJECTED
    assert not unasked, f"read by the template but never asked: {sorted(unasked)}"


def test_hidden_answers_are_never_needed():
    """`show_if` may hide a question and TrueNAS need not pass its value: the render must still work.

    Every scenario renders in strict mode (a missing name is an error, hidden answers are removed),
    so this covers CPU mode (no GPU id), ixVolume (no host path) and each optional service.
    """
    scenarios = [{}, {"use_gpu": "false"}, {"use_gpu": "false", "enable_whisper_cpp": "true"}, OPTIONAL_ON,
                 {**OPTIONAL_ON, "expose_backend_ports": "true"}, {"use_gpu": "false", "expose_backend_ports": "true"}]
    for scenario in scenarios:
        assert render(**scenario)["services"], scenario
    ix = rc.default_answers(questions(), {"storage_type": "ix_volume", "ix_volumes": {"data": DATA + "-ix"}})
    assert yaml.safe_load(rc.render(ix, questions=questions()))["services"]
    assert "host_path" not in rc.visible_answers(questions(), ix)
    cpu = rc.default_answers(questions(), {"host_path": DATA, "use_gpu": "false"})
    assert "gpu_device_id" not in rc.visible_answers(questions(), cpu)


# --- the rendered stack is valid Compose ----------------------------------------------------------------------


@needs_docker
@pytest.mark.parametrize(
    "scenario",
    [{}, {"use_gpu": "false"}, {"use_gpu": "false", "enable_whisper_cpp": "true"}, OPTIONAL_ON,
     {**OPTIONAL_ON, "expose_backend_ports": "true", "frontend_port": 8080},
     {"image_tag": "latest", "pull_policy": "always"}],
    ids=["defaults", "cpu", "cpu-whisper-cpp", "everything", "everything-published-port-8080", "latest"],
)
def test_the_rendered_template_passes_docker_compose_config(scenario):
    text = rc.render(answers(**scenario), questions=questions())
    result = subprocess.run([DOCKER, "compose", "-f", "-", "config", "-q"], input=text, capture_output=True,
                            text=True, timeout=120)
    assert result.returncode == 0, result.stderr


@needs_docker
def test_ixvolume_paths_are_accepted_as_a_string_or_a_mapping():
    for injected in (DATA + "-ix", {"host_path": DATA + "-ix"}):
        ix = rc.default_answers(questions(), {"storage_type": "ix_volume", "ix_volumes": {"data": injected}})
        config = compose_config(rc.render(ix, questions=questions()))
        sources = {v["source"] for s in config["services"].values() for v in s.get("volumes", [])}
        assert sources and all(src.startswith(DATA + "-ix/") for src in sources), sources


@needs_docker
def test_free_text_answers_survive_yaml_and_compose_interpolation():
    """An API key may hold quotes, backslashes, # and $: YAML must not mangle it, Compose must not expand it."""
    key = 'p"a\\ss$word#with: {braces} and $HOME and ${X}'
    origins = "https://voice.example.com,https://other.example.com"
    hosts = "voice.example.com,*.tail1234.ts.net"
    config = compose_config(rc.render(answers(api_key=key, trusted_origins=origins, allowed_origins=origins,
                                              trusted_hosts=hosts),
                                      questions=questions()))
    env = config["services"]["frontend-service"]["environment"]
    # `docker compose config` prints a literal $ as $$ so that its output can be fed back in.
    assert env["API_KEY"].replace("$$", "$") == key
    assert env["TRUSTED_ORIGINS"] == origins and env["ALLOWED_ORIGINS"] == origins
    assert env["TRUSTED_HOSTS"] == hosts


def test_free_text_survives_without_the_docker_cli_too():
    key = 'a"b\\c$d#e:f'
    env = env_mapping(render(api_key=key)["services"]["frontend-service"])
    assert interpolate(str(env["API_KEY"])) == key


# --- what the answers do -------------------------------------------------------------------------------------------


def test_by_default_only_the_web_ui_port_is_published():
    published = {n: s["ports"] for n, s in render()["services"].items() if s.get("ports")}
    assert published == {"frontend-service": ["0.0.0.0:3000:3000"]}


def test_the_web_ui_port_answer_moves_the_mapping_and_the_portal():
    document = render(frontend_port=8080)
    assert document["services"]["frontend-service"]["ports"] == ["0.0.0.0:8080:3000"]
    assert document["x-portals"] == [{"name": "Web UI", "scheme": "http", "host": "0.0.0.0", "port": 8080, "path": "/"}]


def test_publishing_backend_ports_binds_them_to_loopback_only():
    services = render(**OPTIONAL_ON, expose_backend_ports="true")["services"]
    for name, service in services.items():
        for mapping in service.get("ports", []):
            if name != "frontend-service":
                assert mapping.startswith("127.0.0.1:"), f"{name} publishes {mapping} beyond loopback"
    assert sum(1 for n, s in services.items() if n != "frontend-service" and s.get("ports")) >= 8


def test_the_release_and_pull_policy_answers_reach_every_image():
    services = render(**OPTIONAL_ON, image_tag="latest", pull_policy="always", image_registry="registry.example/me")["services"]
    assert {s["image"].rsplit(":", 1)[1] for s in services.values()} == {"latest"}
    assert {s["image"].split("/tts-stt-")[0] for s in services.values()} == {"registry.example/me"}
    assert {s["pull_policy"] for s in services.values()} == {"always"}


def test_optional_services_appear_only_when_asked_for():
    assert set(render()["services"]) == {
        "piper-voices-seed", "frontend-service", "piper-tts-service", "stt-service", "qwen3-asr-service", "qwen3-tts-service"}
    for flag, service in [("enable_canary", "canary-asr-service"), ("enable_parakeet", "parakeet-asr-service"),
                          ("enable_chatterbox", "chatterbox-tts-service"), ("enable_training", "piper-training-service"),
                          ("enable_whisper_cpp", "whisper-cpp")]:
        assert service in render(**{flag: "true"})["services"], flag


def test_the_ui_flags_follow_the_services_that_run():
    env = env_mapping(render(**OPTIONAL_ON)["services"]["frontend-service"])
    assert {k: env[k] for k in env if k.startswith("ENABLE_")} == {
        "ENABLE_PARAKEET_ASR": "true", "ENABLE_CANARY_ASR": "true", "ENABLE_CHATTERBOX_TTS": "true",
        "ENABLE_WHISPER_CPP": "true"}
    off = env_mapping(render()["services"]["frontend-service"])
    assert {off[k] for k in off if k.startswith("ENABLE_")} == {"false"}, "a flag without its service is a red indicator"


def test_cpu_mode_starts_no_gpu_service_and_reserves_no_device():
    document = render(use_gpu="false", **{k: v for k, v in OPTIONAL_ON.items() if k != "enable_whisper_cpp"})
    assert set(document["services"]) == {"piper-voices-seed", "frontend-service", "piper-tts-service", "stt-service"}
    stt = document["services"]["stt-service"]
    assert "deploy" not in stt
    env = env_mapping(stt)
    assert (env["USE_CUDA"], env["FORCE_ACCELERATION"]) == ("false", "cpu")
    assert not [k for k in env if k.startswith("NVIDIA_")]
    frontend = env_mapping(document["services"]["frontend-service"])
    assert frontend["ENABLE_CANARY_ASR"] == frontend["ENABLE_PARAKEET_ASR"] == frontend["ENABLE_CHATTERBOX_TTS"] == "false"


def test_gpu_mode_gives_every_gpu_service_the_selected_device():
    services = render(**OPTIONAL_ON, gpu_device_id="GPU-1234")["services"]
    gpu_services = {n for n, s in services.items() if "deploy" in s}
    assert gpu_services == {"stt-service", "qwen3-asr-service", "qwen3-tts-service", "canary-asr-service",
                            "parakeet-asr-service", "chatterbox-tts-service", "piper-training-service"}
    for name in gpu_services:
        device = services[name]["deploy"]["resources"]["reservations"]["devices"][0]
        assert device == {"driver": "nvidia", "device_ids": ["GPU-1234"], "capabilities": ["gpu"]}, name
        assert services[name]["environment"]["NVIDIA_VISIBLE_DEVICES"] == "GPU-1234"
    assert "deploy" not in services["piper-tts-service"] and "deploy" not in services["whisper-cpp"]


def test_every_host_path_sits_under_the_chosen_dataset():
    for service in render(**OPTIONAL_ON)["services"].values():
        for volume in service.get("volumes", []):
            assert volume.startswith(DATA + "/"), volume


# --- the template against docker-compose.truenas-app.yml ------------------------------------------------------------


def resolve(value, env: dict[str, str]):
    """Compose substitution over a parsed document (``$$`` becomes ``$``)."""
    if isinstance(value, str):
        return interpolate(value.replace("${APP_DATA_DIR:?set dataset path}", env["APP_DATA_DIR"]), env)
    if isinstance(value, list):
        return [resolve(v, env) for v in value]
    if isinstance(value, dict):
        return {k: resolve(v, env) for k, v in value.items()}
    return value


def app_compose(env: dict[str, str] | None = None) -> dict[str, dict]:
    """The Custom App file's services with Compose substitution applied for ``env``."""
    return resolve(load_compose(COMPOSE_APP)["services"], {"APP_DATA_DIR": DATA, **(env or {})})


def normal(value):
    """Compare by meaning: Compose reads every environment value as a string, and null as empty."""
    if isinstance(value, dict):
        return {k: normal(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normal(v) for v in value]
    return "" if value is None else str(value)


# The compose file has things the catalog leaves out on purpose: the notify-only watcher `diun` (a
# service that reads the Docker socket), the labels it reads, and the `profiles` lines that an
# optional service needs in YAML but that are questions in the form.
LEFT_OUT_KEYS = {"labels", "profiles"}
LEFT_OUT_SERVICES = {"diun"}


def differences(scenario: dict, compose_env: dict[str, str], dropped_services: set[str] = frozenset()) -> list[str]:
    rendered = resolve(render(**scenario)["services"], {"APP_DATA_DIR": DATA})
    compose = {n: s for n, s in app_compose(compose_env).items() if n not in LEFT_OUT_SERVICES | dropped_services}
    found = []
    for name in sorted(set(rendered) | set(compose)):
        if name not in rendered:
            found.append(f"{name}: in docker-compose.truenas-app.yml, not rendered by the template")
            continue
        if name not in compose:
            found.append(f"{name}: rendered by the template, not in docker-compose.truenas-app.yml")
            continue
        a, b = compose[name], rendered[name]
        for key in sorted((set(a) | set(b)) - LEFT_OUT_KEYS):
            if key == "environment":
                ea, eb = normal(env_mapping(a)), normal(env_mapping(b))
                for var in sorted(set(ea) | set(eb)):
                    if ea.get(var) != eb.get(var):
                        found.append(f"{name}.environment.{var}: compose {ea.get(var)!r}, template {eb.get(var)!r}")
            elif normal(a.get(key)) != normal(b.get(key)):
                found.append(f"{name}.{key}: compose {normal(a.get(key))!r}, template {normal(b.get(key))!r}")
    return found


UI_FLAGS_ON = {"ENABLE_CANARY_ASR": "true", "ENABLE_PARAKEET_ASR": "true", "ENABLE_CHATTERBOX_TTS": "true",
               "ENABLE_WHISPER_CPP": "true"}


def test_at_the_default_answers_the_template_is_the_default_compose_stack():
    """Same services, images, environment, volumes, health checks, restart and stop behaviour."""
    optional = {n for n, s in load_compose(COMPOSE_APP)["services"].items() if s.get("profiles")}
    assert not differences({}, {}, dropped_services=optional)


def test_with_every_optional_service_on_the_template_is_the_full_compose_stack():
    """The compose file with every `profiles:` line deleted and every UI flag set true."""
    assert not differences(OPTIONAL_ON, UI_FLAGS_ON)


def test_the_comparison_notices_a_difference():
    """Guards the guard: a healthcheck or an environment value that drifts must show up."""
    fake = app_compose({"WHISPER_MODEL_SIZE": "tiny"})
    assert fake["stt-service"]["environment"]["WHISPER_MODEL_SIZE"] == "tiny"
    assert differences({}, {"WHISPER_MODEL_SIZE": "tiny", "STT_DEFAULT_LANGUAGE": "en"},
                       dropped_services={n for n, s in load_compose(COMPOSE_APP)["services"].items() if s.get("profiles")})


def test_the_custom_app_file_settings_and_the_form_agree_on_the_defaults_they_share():
    """The form's defaults are the compose file's ${NAME:-default} for the settings both expose."""
    by_name = {q["variable"]: q["schema"].get("default") for q in questions()}
    stt = env_mapping(app_compose()["stt-service"])
    frontend = env_mapping(app_compose()["frontend-service"])
    qwen_tts = env_mapping(app_compose()["qwen3-tts-service"])
    qwen_asr = env_mapping(app_compose()["qwen3-asr-service"])
    assert by_name["whisper_model_size"] == stt["WHISPER_MODEL_SIZE"]
    assert by_name["stt_default_language"] == stt["STT_DEFAULT_LANGUAGE"]
    assert by_name["qwen3_tts_model"] == qwen_tts["QWEN3_TTS_MODEL"]
    assert by_name["qwen3_asr_model"] == qwen_asr["QWEN3_ASR_MODEL"]
    assert str(by_name["model_ttl"]) == stt["MODEL_TTL"] == stt["STT_MODEL_TTL"]
    assert by_name["allowed_origins"] == frontend["ALLOWED_ORIGINS"]
    assert by_name["trusted_origins"] == frontend["TRUSTED_ORIGINS"]
    assert by_name["trusted_hosts"] == frontend["TRUSTED_HOSTS"]
    assert by_name["api_key"] == frontend["API_KEY"]
    assert by_name["image_registry"] == "ghcr.io/steffenhebestreit"
    assert by_name["frontend_port"] == 3000


# --- the renderer itself --------------------------------------------------------------------------------------------


def run_render(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "truenas" / "render_catalog.py"), *args],
                          capture_output=True, text=True, timeout=60)


def test_the_renderer_command_prints_valid_yaml():
    result = run_render("--set", f"host_path={DATA}", "--set", "enable_canary=true")
    assert result.returncode == 0, result.stderr
    assert "canary-asr-service" in yaml.safe_load(result.stdout)["services"]


@pytest.mark.parametrize(
    "args, message",
    [
        ([], "no default: pass --set host_path"),
        (["--set", f"host_path={DATA}", "--set", "nonsense=1"], "no such question"),
        (["--set", f"host_path={DATA}", "--set", "use_gpu=maybe"], "not true or false"),
        (["--set", f"host_path={DATA}", "--set", "frontend_port=abc"], "not an integer"),
        (["--set", "host_path"], "name=value"),
    ],
)
def test_the_renderer_refuses_what_it_cannot_render(args, message):
    result = run_render(*args)
    assert result.returncode != 0 and message in result.stderr, result.stderr
    assert result.stdout == ""


def test_the_renderer_fails_on_a_template_that_reads_a_name_nobody_asked(tmp_path):
    """StrictUndefined is the point: an empty string in its place would render a broken stack silently."""
    template = CATALOG / "templates" / "docker-compose.yaml"
    broken = tmp_path / "docker-compose.yaml"
    broken.write_text(template.read_text(encoding="utf-8").replace("values.frontend_port", "values.frontend_prot", 1),
                      encoding="utf-8")
    with pytest.raises(jinja2.UndefinedError):
        rc.render(answers(), questions=questions(), template=broken)
