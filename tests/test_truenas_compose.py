"""docker-compose.truenas-app.yml as a person on a NAS meets it: paste, change one path, deploy.

Compose itself is the judge wherever the docker CLI exists (it needs no daemon for `config`), so
what is asserted is what `docker compose config` resolves, not what the YAML text says: the
settings block really is the only place to edit, the placeholder path is refused, only the web UI
port is published, every service that downloads models waits long enough before it is called
unhealthy, and one directory holds everything that must survive an update.

The pure-YAML checks run without the CLI; the ones that need it skip when it is missing, and the
CI workflow (.github/workflows/truenas.yml) has it.
"""

from __future__ import annotations

import ast
import json
import os
import posixpath
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from compose_helpers import REPO_ROOT, dotenv, interpolate, load_compose, require_yaml

yaml = require_yaml()

COMPOSE_APP = REPO_ROOT / "docker-compose.truenas-app.yml"
COMPOSE_OVERLAY = REPO_ROOT / "docker-compose.truenas.yml"
ENV_EXAMPLE = REPO_ROOT / ".env.truenas.example"
ENV_PRODUCTION = REPO_ROOT / ".env.truenas.production.example"
INCLUDE_WRAPPER = REPO_ROOT / "truenas" / "custom-app-include.yml"
DOCKER = shutil.which("docker")
DATA = "/mnt/tank/apps/tts-stt"
PLACEHOLDER = "${APP_DATA_DIR:?set dataset path}"

OPTIONAL_PROFILES = ["canary-asr", "parakeet-asr", "chatterbox-tts", "magpie-tts", "training", "whisper-cpp",
                     "update-watch"]

needs_docker = pytest.mark.skipif(DOCKER is None, reason="docker CLI not found: `docker compose config` cannot run")


def compose(*args: str, env: dict | None = None, stdin: str | None = None) -> subprocess.CompletedProcess:
    full_env = {k: v for k, v in os.environ.items() if k not in ("IMAGE_TAG", "PULL_POLICY", "APP_DATA_DIR", "COMPOSE_PROFILES")}
    full_env.update(env or {})
    return subprocess.run([DOCKER, "compose", *args], capture_output=True, text=True, timeout=180,
                          env=full_env, input=stdin, cwd=str(REPO_ROOT))


def resolved(*args: str, env: dict | None = None) -> dict:
    """`docker compose config --format json` (raises with Compose's own message when it is invalid)."""
    result = compose(*args, "config", "--format", "json", env=env)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def all_profiles() -> list[str]:
    return [a for p in OPTIONAL_PROFILES for a in ("--profile", p)]


def services() -> dict:
    return load_compose(COMPOSE_APP)["services"]


# --- paste, change one path, deploy ------------------------------------------------------------------------------


@needs_docker
def test_pasted_unchanged_the_file_is_refused_with_a_message_that_names_the_setting():
    result = compose("-f", str(COMPOSE_APP), "config", "-q")
    assert result.returncode != 0
    assert "required variable APP_DATA_DIR is missing a value" in result.stderr
    assert "set dataset path" in result.stderr


@needs_docker
def test_the_path_is_the_only_thing_to_change_and_the_stack_then_validates():
    text = COMPOSE_APP.read_text(encoding="utf-8").replace(PLACEHOLDER, DATA)
    assert PLACEHOLDER not in text
    result = compose("-f", "-", "config", "-q", stdin=text)
    assert result.returncode == 0, result.stderr


@needs_docker
@pytest.mark.parametrize("env_file", [ENV_EXAMPLE, ENV_PRODUCTION], ids=lambda p: p.name)
def test_the_file_validates_with_each_env_example_alone_and_with_every_optional_service(env_file):
    for extra in ([], all_profiles()):
        result = compose("--env-file", str(env_file), *extra, "-f", str(COMPOSE_APP), "config", "-q")
        assert result.returncode == 0, (extra, result.stderr)


@needs_docker
def test_the_build_from_source_overlay_validates_with_the_truenas_env_example():
    result = compose("--env-file", str(ENV_EXAMPLE), "--profile", "all", *all_profiles(),
                     "-f", "docker-compose.yml", "-f", str(COMPOSE_OVERLAY), "config", "-q")
    assert result.returncode == 0, result.stderr


@needs_docker
def test_the_overlay_pins_every_image_of_the_base_stack_to_the_release():
    """Base compose defaults to `latest`; the overlay is what makes the build-from-source route pinned."""
    release = yaml.safe_load((REPO_ROOT / "truenas" / "tts-stt" / "app.yaml").read_text(encoding="utf-8"))["app_version"]
    config = resolved("--env-file", str(ENV_EXAMPLE), "--profile", "all", *all_profiles(),
                      "-f", "docker-compose.yml", "-f", str(COMPOSE_OVERLAY))
    images = {n: s["image"] for n, s in config["services"].items() if "image" in s}
    assert len(images) >= 11, images
    floating = {n: i for n, i in images.items() if not i.endswith(f":{release}")}
    assert not floating, f"not on {release}: {floating}"


@needs_docker
def test_the_settings_file_wrapper_reads_its_release_from_the_settings_file(tmp_path):
    """`include:` with `env_file:`, as the guide describes: the pasted YAML stays fixed, the release
    is one line in settings.env. Compose is what TrueNAS runs to validate a pasted file."""
    (tmp_path / "docker-compose.truenas-app.yml").write_text(COMPOSE_APP.read_text(encoding="utf-8"), encoding="utf-8")
    env = ENV_EXAMPLE.read_text(encoding="utf-8")
    env = re.sub(r"^IMAGE_TAG=.*$", "IMAGE_TAG=7.7.7", env, flags=re.M)
    env = re.sub(r"^APP_DATA_DIR=.*$", f"APP_DATA_DIR={DATA}", env, flags=re.M)
    (tmp_path / "settings.env").write_text(env, encoding="utf-8")
    wrapper = INCLUDE_WRAPPER.read_text(encoding="utf-8")
    wrapper = wrapper.replace("/mnt/tank/apps/tts-stt/app/", f"{tmp_path}/")
    (tmp_path / "wrapper.yml").write_text(wrapper, encoding="utf-8")

    config = resolved("-f", str(tmp_path / "wrapper.yml"))
    tags = {s["image"].rsplit(":", 1)[1] for s in config["services"].values()}
    assert tags == {"7.7.7"}, "the tag in settings.env must win over the default in the compose file"
    assert all(v["source"].startswith(DATA) for s in config["services"].values() for v in s.get("volumes", []))


def test_the_settings_file_wrapper_names_the_two_files_the_guide_tells_people_to_download():
    document = yaml.safe_load(INCLUDE_WRAPPER.read_text(encoding="utf-8"))
    [entry] = document["include"]
    assert entry["path"].endswith("/docker-compose.truenas-app.yml")
    assert entry["env_file"].endswith("/settings.env")
    assert document["x-portals"][0]["port"] == 3000


# --- what runs and what is reachable ----------------------------------------------------------------------------------


@needs_docker
def test_only_the_web_ui_port_is_published_whatever_is_enabled():
    config = resolved("--env-file", str(ENV_EXAMPLE), *all_profiles(), "-f", str(COMPOSE_APP))
    published = {n: [(p.get("host_ip"), p["published"], p["target"]) for p in s["ports"]]
                 for n, s in config["services"].items() if s.get("ports")}
    assert published == {"frontend-service": [("0.0.0.0", "3000", 3000)]}, published


@needs_docker
def test_the_web_ui_port_and_bind_address_are_settings():
    config = resolved("-f", str(COMPOSE_APP), env={"APP_DATA_DIR": DATA, "FRONTEND_PORT": "8443", "FRONTEND_BIND_ADDR": "127.0.0.1"})
    [port] = config["services"]["frontend-service"]["ports"]
    assert (port["host_ip"], port["published"], port["target"]) == ("127.0.0.1", "8443", 3000)


@needs_docker
def test_the_default_stack_and_what_a_profile_adds():
    default = resolved("--env-file", str(ENV_EXAMPLE), "-f", str(COMPOSE_APP))["services"]
    assert set(default) == {"piper-voices-seed", "frontend-service", "piper-tts-service", "stt-service",
                            "qwen3-asr-service", "qwen3-tts-service"}
    everything = resolved("--env-file", str(ENV_EXAMPLE), *all_profiles(), "-f", str(COMPOSE_APP))["services"]
    assert set(everything) - set(default) == {"canary-asr-service", "parakeet-asr-service", "chatterbox-tts-service",
                                              "magpie-tts-service", "piper-training-service", "whisper-cpp", "diun"}


@needs_docker
def test_a_deleted_service_leaves_the_rest_valid():
    """The guide tells people with no GPU to delete stt/qwen services: nothing may hard-depend on them."""
    document = load_compose(COMPOSE_APP)
    for gone in ("stt-service", "qwen3-asr-service", "qwen3-tts-service"):
        del document["services"][gone]
    for service in document["services"].values():
        service.get("depends_on", {}) and [service["depends_on"].pop(g, None) for g in
                                           ("stt-service", "qwen3-asr-service", "qwen3-tts-service")]
    result = compose("-f", "-", "config", "-q", stdin=yaml.safe_dump(document), env={"APP_DATA_DIR": DATA})
    assert result.returncode == 0, result.stderr


# --- the seed job, run for real ----------------------------------------------------------------------------------------


def run_seed(tmp_path: Path, image_voices: dict[str, str], dataset_voices: dict[str, str]) -> subprocess.CompletedProcess:
    """Run the seed job's own script (from the compose file) against temporary directories.

    Compose turns `$$` into `$`; the two fixed paths in the script are pointed at temp directories.
    """
    script = interpolate(services()["piper-voices-seed"]["command"][0])
    image, models = tmp_path / "image-default", tmp_path / "models"
    image.mkdir(parents=True, exist_ok=True)
    (models / "default").mkdir(parents=True, exist_ok=True)
    for name, content in image_voices.items():
        (image / name).write_text(content, encoding="utf-8")
    for name, content in dataset_voices.items():
        (models / "default" / name).write_text(content, encoding="utf-8")
    script = script.replace("/app/models/default", str(image)).replace("/seed/default", str(models / "default"))
    shell = shutil.which("sh")
    return subprocess.run([shell, "-c", script], capture_output=True, text=True, timeout=60)


@pytest.mark.skipif(shutil.which("sh") is None, reason="no POSIX shell")
def test_the_seed_job_copies_missing_voices_and_never_overwrites_the_users_files(tmp_path):
    """The models directory is a bind mount that hides the voices baked into the image, so a first
    start copies them in. A voice the user replaced or trained must survive every later update."""
    result = run_seed(tmp_path, {"a.onnx": "image a", "a.onnx.json": "{}", "b.onnx": "image b"},
                      {"a.onnx": "the user's own a"})
    target = tmp_path / "models" / "default"
    assert result.returncode == 0, result.stderr
    assert "2 new voice file(s)" in result.stdout
    assert (target / "a.onnx").read_text(encoding="utf-8") == "the user's own a"
    assert (target / "b.onnx").read_text(encoding="utf-8") == "image b"
    assert (target / "a.onnx.json").exists()
    assert not list(target.glob("*.part")), "a copy is finished by a rename, so a crash leaves no half voice"


@pytest.mark.skipif(shutil.which("sh") is None, reason="no POSIX shell")
def test_the_seed_job_is_idempotent_and_succeeds_when_there_is_nothing_to_copy(tmp_path):
    first = run_seed(tmp_path, {"a.onnx": "image a"}, {})
    second = run_seed(tmp_path, {"a.onnx": "image a"}, {})
    assert "1 new voice file(s)" in first.stdout and "0 new voice file(s)" in second.stdout
    assert first.returncode == second.returncode == 0
    empty = run_seed(tmp_path / "other", {}, {})
    assert empty.returncode == 0 and "0 new voice file(s)" in empty.stdout, "exit 0 is a normal state for a TrueNAS app"
    no_dir = run_seed(tmp_path / "third", {}, {})
    shutil.rmtree(tmp_path / "third" / "image-default")
    script = interpolate(services()["piper-voices-seed"]["command"][0])
    script = script.replace("/app/models/default", str(tmp_path / "third" / "image-default"))
    script = script.replace("/seed/default", str(tmp_path / "third" / "models" / "default"))
    missing = subprocess.run([shutil.which("sh"), "-c", script], capture_output=True, text=True, timeout=60)
    assert no_dir.returncode == 0 and missing.returncode == 0, "an image without bundled voices must not fail the app"


# --- the settings files ------------------------------------------------------------------------------------------------


def referenced_variables(path: Path) -> set[str]:
    """Names substituted by ${NAME...} in the values of a compose file (comments do not count)."""
    found: set[str] = set()

    def walk(node):
        if isinstance(node, str):
            found.update(re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)", node))
        elif isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, dict):
            for key, value in node.items():
                walk(key)
                walk(value)

    walk(load_compose(path))
    return found


@pytest.mark.parametrize("env_file", [ENV_EXAMPLE, ENV_PRODUCTION], ids=lambda p: p.name)
def test_every_setting_an_env_example_switches_on_is_read_by_a_compose_file(env_file):
    """A dead setting (renamed in compose, left in the example) changes nothing and nobody notices.

    The settings-file route feeds the example to docker-compose.truenas-app.yml; the build route
    feeds it to docker-compose.yml plus the overlay. A variable has to be read by one of them.
    """
    readers = (referenced_variables(COMPOSE_APP) | referenced_variables(COMPOSE_OVERLAY)
               | referenced_variables(REPO_ROOT / "docker-compose.yml"))
    dead = sorted(set(dotenv(env_file)) - readers)
    assert not dead, f"{env_file.name} sets variables no compose file reads: {dead}"


@pytest.mark.parametrize("env_file", [ENV_EXAMPLE, ENV_PRODUCTION], ids=lambda p: p.name)
def test_the_settings_file_route_reads_every_setting_it_shows_from_the_custom_app_file(env_file):
    """The wrapper feeds the example only to the Custom App file, so a setting that file ignores
    would be silently dead on that route. Build-route-only settings must stay commented out."""
    unread = sorted(set(dotenv(env_file)) - referenced_variables(COMPOSE_APP))
    assert not unread, f"{env_file.name}: the Custom App file does not read {unread}"


# --- persistence, permissions -------------------------------------------------------------------------------------------


def volume_source(volume: str) -> str:
    """Host side of a short-form volume, with the path setting substituted (its `:?` contains a colon)."""
    return volume.replace(PLACEHOLDER, DATA).split(":")[0]



def test_every_persistent_path_is_a_bind_mount_under_the_one_dataset_and_nothing_uses_a_named_volume():
    document = load_compose(COMPOSE_APP)
    assert not document.get("volumes"), "a named volume would live in the Docker root, outside the dataset"
    for name, service in document["services"].items():
        for volume in service.get("volumes", []):
            source = volume_source(volume)
            if name == "diun" and source == "/var/run/docker.sock":
                continue
            assert source.startswith(DATA + "/") and PLACEHOLDER in volume, f"{name}: {volume} is outside APP_DATA_DIR"


def test_one_model_cache_is_shared_by_every_service_that_downloads_models():
    cache_users = {n for n, s in services().items() if any(v.endswith(":/root/.cache") for v in s.get("volumes", []))}
    assert cache_users == {"stt-service", "qwen3-asr-service", "qwen3-tts-service", "canary-asr-service",
                           "parakeet-asr-service", "chatterbox-tts-service", "magpie-tts-service"}
    sources = {volume_source(v) for n in cache_users for v in services()[n]["volumes"] if v.endswith(":/root/.cache")}
    assert sources == {f"{DATA}/cache"}, "a model would download once per service"


# --- the settings folders ----------------------------------------------------------------------------------------------
#
# The Settings page (web UI -> gear) saves into the dataset's settings/ folder, mounted into the gateway only.
# backend-settings/ is the folder later releases save backend settings to: the gateway writes it, every Python
# backend mounts the whole folder read-only. A container that could write a folder another one mounts (or a parent
# of it) could swap that folder for a symlink before the other one starts, so no mount source may lie inside another
# service's read-write mount. Checked in this file and in docker-compose.yml, which mounts the same two folders.

BASE_COMPOSE = REPO_ROOT / "docker-compose.yml"
DEPLOYMENT_FILES = [COMPOSE_APP, BASE_COMPOSE]
PYTHON_BACKENDS = {"piper-tts-service", "piper-training-service", "stt-service", "qwen3-asr-service",
                   "qwen3-tts-service", "parakeet-asr-service", "canary-asr-service", "chatterbox-tts-service",
                   "magpie-tts-service"}


def short_mounts(path: Path) -> dict[str, list[tuple[str, str, str]]]:
    """{service: [(host source, container target, "ro" or "rw")]}, the dataset path filled in.

    A named volume keeps its name as the source (it does not start with a slash).
    """
    out = {}
    for name, service in load_compose(path)["services"].items():
        entries = []
        for volume in service.get("volumes", []):
            text = interpolate(str(volume).replace(PLACEHOLDER, DATA), {"APP_DATA_DIR": DATA})
            source, target, *options = text.split(":")
            mode = "ro" if options and "ro" in options[0].split(",") else "rw"
            entries.append((posixpath.normpath(source) if source.startswith("/") else source, target, mode))
        out[name] = entries
    return out


def inside(path: str, folder: str) -> bool:
    """Is ``path`` the folder itself or anything below it?"""
    return path == folder or path.startswith(folder.rstrip("/") + "/")


def containment_problems(mounts: dict[str, list[tuple[str, str, str]]]) -> list[str]:
    """Host paths one service mounts that lie strictly inside a folder another service mounts read-write."""
    problems = []
    for writer, writes in mounts.items():
        for folder, _, mode in writes:
            if mode != "rw" or not folder.startswith("/"):
                continue
            for reader, reads in mounts.items():
                for source, target, _ in reads:
                    if reader != writer and source != folder and inside(source, folder):
                        problems.append(f"{reader} mounts {source} (at {target}), inside {writer}'s "
                                        f"read-write {folder}")
    return problems


def test_the_python_backends_are_the_services_built_from_this_repository():
    """The premise of the checks below, so a new backend cannot slip past them unnoticed."""
    for path in DEPLOYMENT_FILES:
        built_here = {name for name in load_compose(path)["services"]
                      if name != "frontend-service" and (REPO_ROOT / name / "app.py").is_file()}
        assert built_here == PYTHON_BACKENDS, (path.name, sorted(built_here ^ PYTHON_BACKENDS))


@pytest.mark.parametrize("path", DEPLOYMENT_FILES, ids=lambda p: p.name)
def test_the_gateway_mounts_both_settings_folders_read_write(path):
    gateway = {target: (source, mode) for source, target, mode in short_mounts(path)["frontend-service"]}
    assert gateway.get("/app/settings") == (f"{DATA}/settings", "rw"), gateway
    assert gateway.get("/app/backend-settings") == (f"{DATA}/backend-settings", "rw"), gateway


@pytest.mark.parametrize("path", DEPLOYMENT_FILES, ids=lambda p: p.name)
def test_every_python_backend_mounts_the_whole_backend_settings_folder_read_only(path):
    mounts = short_mounts(path)
    for name in sorted(PYTHON_BACKENDS):
        found = [(source, mode) for source, target, mode in mounts[name] if target == "/app/backend-settings"]
        assert found == [(f"{DATA}/backend-settings", "ro")], f"{name}: {found}"
        assert not [s for s, t, _ in mounts[name]
                    if inside(s, f"{DATA}/backend-settings") and t != "/app/backend-settings"], name
    for name in sorted(set(mounts) - PYTHON_BACKENDS - {"frontend-service"}):
        assert not [s for s, _, _ in mounts[name] if inside(s, f"{DATA}/backend-settings")], (
            f"{name} is not a Python backend and has no business with backend-settings")


@pytest.mark.parametrize("path", DEPLOYMENT_FILES, ids=lambda p: p.name)
def test_only_the_gateway_mounts_the_settings_folder(path):
    """It holds the API key digests and the audit log: no other container reads or writes it."""
    for name, entries in short_mounts(path).items():
        if name == "frontend-service":
            continue
        assert not [(s, t) for s, t, _ in entries if inside(s, f"{DATA}/settings") or inside(t, "/app/settings")], name


@pytest.mark.parametrize("path", DEPLOYMENT_FILES, ids=lambda p: p.name)
def test_no_mount_source_lies_inside_another_services_read_write_mount(path):
    assert not containment_problems(short_mounts(path))


def test_the_containment_check_notices_a_folder_mounted_inside_another_ones():
    """Guards the guard: a backend that mounted a file of settings/, or the gateway mounting the parent of models/."""
    planted = short_mounts(COMPOSE_APP)
    planted["stt-service"].append((f"{DATA}/settings/keys.json", "/tmp/keys.json", "ro"))
    assert containment_problems(planted) == [
        f"stt-service mounts {DATA}/settings/keys.json (at /tmp/keys.json), inside frontend-service's "
        f"read-write {DATA}/settings"]
    planted = short_mounts(COMPOSE_APP)
    planted["frontend-service"].append((DATA, "/data", "rw"))
    assert len(containment_problems(planted)) > 10
    assert not containment_problems({"a": [(f"{DATA}/settings-old", "/x", "rw")], "b": [(f"{DATA}/settings", "/y", "ro")]})


def mount_lines_the_settings_page_shows() -> dict[str, list[str]]:
    """frontend-service/app.py's _SETTINGS_MOUNT_LINES, read without importing the gateway."""
    tree = ast.parse((REPO_ROOT / "frontend-service" / "app.py").read_text(encoding="utf-8"))
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if any(getattr(target, "id", None) == "_SETTINGS_MOUNT_LINES" for target in targets):
            return ast.literal_eval(node.value)
    raise AssertionError("frontend-service/app.py no longer defines _SETTINGS_MOUNT_LINES")


def test_the_lines_the_settings_page_shows_without_a_mount_are_the_lines_of_the_files():
    """An install pasted before the Settings page has no mount: the page shows what to add. It must be this."""
    shown = mount_lines_the_settings_page_shows()
    for key, path in (("truenas", COMPOSE_APP), ("compose", BASE_COMPOSE)):
        lines = [yaml.safe_load(line)[0] for line in shown[key]]
        assert lines == load_compose(path)["services"]["frontend-service"]["volumes"], (key, path.name)


def test_the_containers_run_as_root_which_is_what_the_guide_says_about_permissions():
    """The guide: 'every container runs as root, so no chown or ACL is needed'. True only while no
    compose service sets `user:` and no image switches user; this fails the day one does."""
    for name, service in services().items():
        assert "user" not in service, f"{name} sets user: the guide's permission advice no longer holds"
    for dockerfile in REPO_ROOT.glob("*/Dockerfile"):
        users = re.findall(r"^USER\s+(\S+)", dockerfile.read_text(encoding="utf-8"), flags=re.M)
        assert all(u in ("root", "0") for u in users), f"{dockerfile.parent.name} runs as {users}"


# --- health, restarts, stopping -----------------------------------------------------------------------------------------------


def healthchecks() -> dict[str, dict]:
    return {n: s["healthcheck"] for n, s in services().items() if "healthcheck" in s}


def seconds(text: str) -> int:
    match = re.fullmatch(r"(\d+)([sm])", str(text))
    return int(match.group(1)) * (60 if match.group(2) == "m" else 1)


def test_every_long_running_service_has_a_liveness_check_on_health():
    checks = healthchecks()
    long_running = set(services()) - {"piper-voices-seed"}
    assert set(checks) == long_running
    for name, check in checks.items():
        probe = " ".join(check["test"])
        if name == "diun":
            assert check["test"] == ["CMD", "diun", "healthcheck"]
        elif name == "whisper-cpp":
            assert "localhost:8080/" in probe, "whisper-server answers any path; any HTTP answer is alive"
        else:
            assert re.search(r"http://localhost:\d+/health\b", probe), f"{name}: {probe}"
            assert "/ready" not in probe, f"{name}: readiness would keep the app 'Deploying' for minutes"


def test_services_that_download_models_wait_long_enough_before_they_can_be_unhealthy():
    """First start pulls gigabytes; a start_period shorter than that shows a red app for a healthy install."""
    for name in ("stt-service", "qwen3-asr-service", "qwen3-tts-service", "whisper-cpp"):
        assert seconds(healthchecks()[name]["start_period"]) >= 600, name
    for name in ("canary-asr-service", "parakeet-asr-service", "chatterbox-tts-service", "magpie-tts-service"):
        assert seconds(healthchecks()[name]["start_period"]) >= 900, name
    assert seconds(healthchecks()["frontend-service"]["start_period"]) >= 30


def test_every_service_restarts_unless_stopped_except_the_one_shot_seed_job():
    for name, service in services().items():
        expected = "no" if name == "piper-voices-seed" else "unless-stopped"
        assert service.get("restart") == expected, name


def test_every_service_has_a_stop_grace_period_and_training_gets_longest():
    graces = {}
    for name, service in services().items():
        if name == "piper-voices-seed":
            continue
        assert "stop_grace_period" in service, f"{name}: Docker's 10 s default would SIGKILL a running job"
        graces[name] = seconds(service["stop_grace_period"])
    assert graces["piper-training-service"] == max(graces.values())
    training = services()["piper-training-service"]["environment"]["TRAINING_SHUTDOWN_GRACE_S"]
    assert graces["piper-training-service"] > int(interpolate(training)), "SIGKILL must not come before the jobs stop"


def test_the_seed_job_completes_before_piper_starts_and_uses_no_network():
    document = services()
    assert document["piper-tts-service"]["depends_on"]["piper-voices-seed"]["condition"] == "service_completed_successfully"
    assert document["piper-voices-seed"]["network_mode"] == "none"


def test_the_gateway_never_waits_for_a_model():
    """It may start before its backends: optional ones are absent and models load for minutes."""
    depends = services()["frontend-service"]["depends_on"]
    assert {spec["condition"] for spec in depends.values()} == {"service_started"}
    assert all(spec.get("required") is False for spec in depends.values())


# --- the settings block ---------------------------------------------------------------------------------------------------


def test_the_settings_named_in_the_header_are_the_ones_the_file_reads():
    """Every setting the header table tells people to search for exists, with the default it shows."""
    text = COMPOSE_APP.read_text(encoding="utf-8")
    header = text.split("services:", 1)[0]
    for needle in ("IMAGE_TAG:-", "PULL_POLICY:-missing", "GPU_DEVICE_ID:-0", "FRONTEND_PORT:-3000", PLACEHOLDER):
        assert needle in header, f"the settings block does not document {needle}"
        assert needle in text[len(header):], f"the settings block documents {needle}, which no service uses"
    assert dotenv(ENV_EXAMPLE)["PULL_POLICY"] == "missing" == interpolate("${PULL_POLICY:-missing}")


def test_the_portal_and_notes_keys_match_what_truenas_accepts():
    """TrueNAS validates x-portals with a JSON schema (name, scheme, host, port required; path
    starting with /; nothing else). Read from the middleware source, not run on a NAS."""
    document = yaml.safe_load(COMPOSE_APP.read_text(encoding="utf-8"))
    for portal in document["x-portals"]:
        assert set(portal) <= {"name", "path", "scheme", "host", "port"}
        assert {"name", "scheme", "host", "port"} <= set(portal)
        assert portal["scheme"] in ("http", "https") and isinstance(portal["port"], int)
        assert portal.get("path", "/").startswith("/")
    assert isinstance(document["x-notes"], str) and document["x-notes"].strip()


def test_a_second_portal_opens_the_settings_page_on_the_web_ui_port():
    document = yaml.safe_load(COMPOSE_APP.read_text(encoding="utf-8"))
    portals = {portal["name"]: portal for portal in document["x-portals"]}
    assert list(portals) == ["Web UI", "Settings"]
    assert (portals["Web UI"]["path"], portals["Settings"]["path"]) == ("/", "/settings")
    assert portals["Settings"]["port"] == portals["Web UI"]["port"] == int(interpolate("${FRONTEND_PORT:-3000}"))
    assert {portals["Settings"][k] for k in ("scheme", "host")} == {portals["Web UI"][k] for k in ("scheme", "host")}
    header = COMPOSE_APP.read_text(encoding="utf-8").split("services:", 1)[0]
    assert "both `port:` lines under x-portals" in header, "the port setting must say there are two portals now"


def test_the_notes_say_where_the_settings_live_and_how_to_get_back_in():
    """Comments are gone after Save; x-notes is what a locked-out owner still has."""
    notes = " ".join(yaml.safe_load(COMPOSE_APP.read_text(encoding="utf-8"))["x-notes"].split())
    for needle in ("/settings", "settings/gateway.json", "settings/SAFE-MODE", "by IP address", "frontend-service log"):
        assert needle in notes, f"x-notes does not mention {needle!r}"
