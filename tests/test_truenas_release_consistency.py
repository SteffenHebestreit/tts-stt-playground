"""One release number for the TrueNAS files, and one rule for keeping it that way.

``app_version`` in truenas/tts-stt/app.yaml is the single source of truth: it is the image tag
every TrueNAS file installs by default (``IMAGE_TAG``). A pinned tag only helps if all the places
that name it agree, so a person who pastes the compose file, copies an env example or fills in the
catalog form gets the same release.

Also checked here: ``latest`` is an opt-in channel that really pulls (``pull_policy: always``), and
scripts/truenas/set_version.py, the tool that moves the release, does what its docstring says on a
copy of the tree.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from compose_helpers import REPO_ROOT, dotenv, interpolate, load_compose, require_yaml

yaml = require_yaml()

APP_YAML = REPO_ROOT / "truenas" / "tts-stt" / "app.yaml"
QUESTIONS = REPO_ROOT / "truenas" / "tts-stt" / "questions.yaml"
COMPOSE_APP = REPO_ROOT / "docker-compose.truenas-app.yml"
COMPOSE_OVERLAY = REPO_ROOT / "docker-compose.truenas.yml"
ENV_EXAMPLES = [REPO_ROOT / ".env.truenas.example", REPO_ROOT / ".env.truenas.production.example"]
SET_VERSION = REPO_ROOT / "scripts" / "truenas" / "set_version.py"
PUBLISH = REPO_ROOT / ".github" / "workflows" / "publish-images.yml"

REGISTRY = "ghcr.io/steffenhebestreit"
SEMVER = re.compile(r"\d+\.\d+\.\d+")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


set_version = _load(SET_VERSION, "truenas_set_version")
render_catalog = _load(REPO_ROOT / "scripts" / "truenas" / "render_catalog.py", "truenas_render_catalog")


def release() -> str:
    return yaml.safe_load(APP_YAML.read_text(encoding="utf-8"))["app_version"]


def our_images(compose: Path, env: dict | None = None) -> dict[str, tuple[str, str]]:
    """service -> (image, pull_policy) for the tts-stt images, with Compose substitution applied."""
    out = {}
    for name, service in load_compose(compose)["services"].items():
        image = interpolate(str(service["image"]), env)
        if image.startswith(f"{REGISTRY}/tts-stt-"):
            out[name] = (image, interpolate(str(service.get("pull_policy", "")), env))
    return out


def tag_of(image: str) -> str:
    return image.rsplit(":", 1)[1]


# --- the source of truth ------------------------------------------------------------------------


def test_the_release_is_a_plain_semantic_version_and_the_catalog_version_follows_it():
    app = yaml.safe_load(APP_YAML.read_text(encoding="utf-8"))
    assert SEMVER.fullmatch(str(app["app_version"])), "a pinned release tag is MAJOR.MINOR.PATCH, not latest"
    assert str(app["version"]) == str(app["app_version"]), "the catalog version follows the app version one to one"


def test_the_custom_app_file_defaults_every_image_to_the_release():
    images = our_images(COMPOSE_APP)
    assert len(images) >= 12, sorted(images)  # 5 default + 6 optional services + the seed job
    wrong = {name: image for name, (image, _) in images.items() if tag_of(image) != release()}
    assert not wrong, f"images that do not default to {release()}: {wrong}"


def test_the_build_from_source_overlay_defaults_every_image_to_the_release():
    images = our_images(COMPOSE_OVERLAY)
    assert images, "the overlay names the images it pins"
    wrong = {name: image for name, (image, _) in images.items() if tag_of(image) != release()}
    assert not wrong, f"images that do not default to {release()}: {wrong}"


def test_every_service_of_the_custom_app_file_is_one_of_our_images_except_the_watcher():
    """A third-party image with a floating tag would undo the pinning; only diun is allowed."""
    services = load_compose(COMPOSE_APP)["services"]
    foreign = {n for n in services if n not in our_images(COMPOSE_APP)}
    assert foreign == {"diun"}, foreign


@pytest.mark.parametrize("path", ENV_EXAMPLES, ids=lambda p: p.name)
def test_the_env_examples_pin_the_release_and_keep_the_pull_policy_matching(path):
    values = dotenv(path)
    assert values["IMAGE_TAG"] == release()
    assert values["PULL_POLICY"] == "missing", "a pinned tag is pulled once; always is for latest"


def test_the_catalog_form_defaults_to_the_release_and_to_pulling_only_what_is_missing():
    questions = {q["variable"]: q for q in yaml.safe_load(QUESTIONS.read_text(encoding="utf-8"))["questions"]}
    assert str(questions["image_tag"]["schema"]["default"]) == release()
    assert questions["pull_policy"]["schema"]["default"] == "missing"


def test_the_catalog_template_renders_the_release_into_every_image():
    answers = render_catalog.default_answers(render_catalog.load_questions(), {
        "host_path": "/mnt/tank/apps/tts-stt", "enable_canary": "true", "enable_parakeet": "true",
        "enable_chatterbox": "true", "enable_magpie": "true", "enable_training": "true",
        "enable_whisper_cpp": "true"})
    rendered = yaml.safe_load(render_catalog.render(answers))
    tags = {n: tag_of(s["image"]) for n, s in rendered["services"].items()}
    assert set(tags.values()) == {release()}, tags


# --- pinned by default, latest on request ----------------------------------------------------------


@pytest.mark.parametrize("compose", [COMPOSE_APP, COMPOSE_OVERLAY], ids=lambda p: p.name)
def test_by_default_nothing_is_pulled_once_it_is_on_the_host(compose):
    """`missing`: a pinned tag never changes, so pulling on every start would only slow Start down."""
    policies = {name: policy for name, (_, policy) in our_images(compose).items() if policy}
    if compose == COMPOSE_APP:
        assert set(policies) == set(our_images(compose)), "every image sets pull_policy explicitly"
    assert set(policies.values()) <= {"missing"}, policies


def test_following_latest_pulls_on_every_start_for_every_image():
    """IMAGE_TAG=latest with PULL_POLICY=always is the documented opt-in: both must reach every service."""
    env = {"IMAGE_TAG": "latest", "PULL_POLICY": "always"}
    images = our_images(COMPOSE_APP, env)
    assert {tag_of(image) for image, _ in images.values()} == {"latest"}
    assert {policy for _, policy in images.values()} == {"always"}


def test_the_nemo_rollback_suffix_is_empty_by_default_so_it_never_changes_the_tag():
    services = load_compose(COMPOSE_APP)["services"]
    for name in ("canary-asr-service", "parakeet-asr-service"):
        raw = str(services[name]["image"])
        assert "${NEMO_IMAGE_SUFFIX:-}" in raw
        assert interpolate(raw, {}).endswith(f":{release()}")
        assert interpolate(raw, {"NEMO_IMAGE_SUFFIX": "-nemo2"}).endswith(f":{release()}-nemo2")


# --- everything that quotes the release --------------------------------------------------------------


def test_every_place_that_quotes_the_release_agrees_with_it():
    """Compose defaults, env examples, catalog form and the commands in the docs (set_version's rule)."""
    problems = set_version.mismatches()
    assert not problems, "\n".join(problems)
    seen = {name for name, *_ in set_version.release_tokens()}
    assert {"docker-compose.truenas-app.yml", "docker-compose.truenas.yml", ".env.truenas.example",
            "docs/truenas-installation-guide.md"} <= seen, f"the rule saw too little: {sorted(seen)}"


PLAIN_RELEASE_TAG = re.compile(r"v\d+\.\d+\.\d+")


@pytest.mark.skipif(
    os.environ.get("GITHUB_REF_TYPE") != "tag" or not PLAIN_RELEASE_TAG.fullmatch(os.environ.get("GITHUB_REF_NAME", "")),
    reason="only applies to a build of a release tag (vX.Y.Z; a prerelease tag never moves a pin)")
def test_a_release_tag_names_the_release_the_files_default_to():
    """publish-images.yml calls the tests before it publishes: a tag that differs from app_version
    would publish images nobody's compose file points at."""
    assert os.environ["GITHUB_REF_NAME"] == f"v{release()}", (
        f"tag {os.environ['GITHUB_REF_NAME']} but app_version is {release()}: run "
        f"scripts/truenas/set_version.py with the tag's version and commit before tagging")


# --- what the publishing workflow has to provide ------------------------------------------------------

plan_publish = _load(REPO_ROOT / "scripts" / "plan_publish.py", "truenas_plan_publish")


def compose_image_names(env: dict | None = None) -> set[str]:
    """The image name suffixes (after tts-stt-, before :tag) the Custom App file pulls."""
    return {image.split("/tts-stt-")[1].rsplit(":", 1)[0] for image, _ in our_images(COMPOSE_APP, env).values()}


def test_the_images_the_compose_file_pulls_are_images_a_release_publishes():
    """A service a release does not build is a pinned install that fails on `manifest unknown`."""
    plan = plan_publish.make_plan("push", f"refs/tags/v{release()}")
    published = {leg["name"] for leg in plan.merge if not leg["suffix"]}
    assert compose_image_names() <= published, sorted(compose_image_names() - published)


def test_the_nemo_rollback_images_the_suffix_selects_are_published_with_the_release():
    """NEMO_IMAGE_SUFFIX=-nemo2 turns canary and parakeet into <release>-nemo2: those must exist."""
    plan = plan_publish.make_plan("push", f"refs/tags/v{release()}")
    rollback = {leg["name"] for leg in plan.merge if leg["suffix"] == "-nemo2"}
    with_suffix = {n for n, s in load_compose(COMPOSE_APP)["services"].items()
                   if "NEMO_IMAGE_SUFFIX" in str(s.get("image"))}
    assert with_suffix == {"canary-asr-service", "parakeet-asr-service"}
    assert rollback == {"canary-asr", "parakeet-asr"}
    assert plan_publish.NEMO2_SUFFIX == "-nemo2", "the compose files spell the suffix -nemo2"


def test_only_the_release_of_a_version_writes_that_version_tag():
    """The pinned tag must not move under an install: not from master, not from a manual run."""
    assert plan_publish.make_plan("push", f"refs/tags/v{release()}").release is True
    assert plan_publish.make_plan("push", "refs/heads/master").release is False
    with pytest.raises(plan_publish.PlanError):
        plan_publish.make_plan("workflow_dispatch", "refs/heads/feature", extra_tag=release())
    with pytest.raises(plan_publish.PlanError):
        plan_publish.make_plan("workflow_dispatch", "refs/heads/feature", extra_tag="latest")


def test_a_release_moves_latest_and_a_prerelease_does_not():
    """`latest` is the opt-in channel of the TrueNAS files: a release candidate must not reach it."""
    assert plan_publish.make_plan("push", f"refs/tags/v{release()}").latest is True
    assert plan_publish.make_plan("push", f"refs/tags/v{release()}-rc.1").latest is False


def test_a_version_tag_publishes_the_bare_release_number_the_compose_file_pins():
    """`git tag v0.2.0` must yield the image tag `0.2.0` (no leading v)."""
    workflow = yaml.safe_load(PUBLISH.read_text(encoding="utf-8"))
    triggers = workflow.get("on", workflow.get(True))
    assert any(str(t).startswith("v") for t in triggers["push"].get("tags", [])), "no v* tag trigger"
    tag_rules = []
    for job in workflow["jobs"].values():
        for step in job.get("steps", []):
            if str(step.get("uses", "")).startswith("docker/metadata-action"):
                tag_rules += str((step.get("with") or {}).get("tags", "")).splitlines()
    rules = [r.strip() for r in tag_rules if r.strip() and not r.strip().startswith("#")]
    assert any(r.startswith("type=semver") and "{{version}}" in r for r in rules), rules


# --- set_version.py: the tool that moves the release ----------------------------------------------------


COPIED = [
    "truenas/tts-stt/app.yaml", "truenas/tts-stt/questions.yaml", "truenas/tts-stt/app-readme.md",
    "truenas/README.md", "truenas/custom-app-include.yml",
    "docker-compose.truenas-app.yml", "docker-compose.truenas.yml",
    ".env.truenas.example", ".env.truenas.production.example",
    "docs/truenas-installation-guide.md", "docs/truenas-deployment.md",
    "docs/truenas-service-profiles.md", "docs/truenas-custom-app-checklist.md",
]


@pytest.fixture
def tree(tmp_path):
    """A copy of the files set_version.py works on."""
    for name in COPIED:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO_ROOT / name, target)
    return tmp_path


def test_a_new_release_reaches_every_place_and_the_check_passes(tree):
    changed = set_version.apply("9.8.7", tree)
    assert set_version.source_version(tree) == "9.8.7"
    assert set_version.mismatches(tree) == []
    assert {"truenas/tts-stt/app.yaml", "truenas/tts-stt/questions.yaml", "docker-compose.truenas-app.yml",
            "docker-compose.truenas.yml", ".env.truenas.example"} <= set(changed)
    # what a user of the changed tree gets
    images = our_images(tree / "docker-compose.truenas-app.yml")
    assert {tag_of(image) for image, _ in images.values()} == {"9.8.7"}
    assert dotenv(tree / ".env.truenas.example")["IMAGE_TAG"] == "9.8.7"
    app = yaml.safe_load((tree / "truenas/tts-stt/app.yaml").read_text(encoding="utf-8"))
    assert (app["version"], app["app_version"]) == ("9.8.7", "9.8.7")


def test_a_second_run_with_the_same_release_changes_nothing(tree):
    set_version.apply("9.8.7", tree)
    assert set_version.apply("9.8.7", tree) == []


def test_only_release_shaped_tokens_move_never_a_bare_number(tree):
    guide = tree / "docs/truenas-installation-guide.md"
    old = release()
    guide.write_text(guide.read_text(encoding="utf-8") + (
        f"\nVersion {old} of an unrelated package, on 2026-{old}, is not a release token,\n"
        f"but `--tag {old}` and IMAGE_TAG:-{old} and `git checkout v{old}` are.\n"), encoding="utf-8")
    set_version.apply("9.8.7", tree)
    tail = guide.read_text(encoding="utf-8").splitlines()[-2:]
    assert f"Version {old} of an unrelated package, on 2026-{old}, is not a release token," == tail[0]
    assert tail[1] == "but `--tag 9.8.7` and IMAGE_TAG:-9.8.7 and `git checkout v9.8.7` are."


def test_latest_is_not_a_release_and_stays(tree):
    compose = tree / "docker-compose.truenas-app.yml"
    compose.write_text(compose.read_text(encoding="utf-8") + "\n# IMAGE_TAG:-latest follows the newest build\n", encoding="utf-8")
    set_version.apply("9.8.7", tree)
    assert "# IMAGE_TAG:-latest follows the newest build" in compose.read_text(encoding="utf-8")


def test_crlf_line_endings_survive(tree):
    env = tree / ".env.truenas.example"
    env.write_bytes(env.read_bytes().replace(b"\n", b"\r\n"))
    set_version.apply("9.8.7", tree)
    data = env.read_bytes()
    assert b"IMAGE_TAG=9.8.7\r\n" in data and b"\r\r" not in data and data.count(b"\n") == data.count(b"\r\n")


def test_a_place_that_disagrees_is_named_with_file_and_line(tree):
    compose = tree / "docker-compose.truenas-app.yml"
    text = compose.read_text(encoding="utf-8")
    compose.write_text(text.replace(f"${{IMAGE_TAG:-{release()}}}", "${IMAGE_TAG:-0.0.1}", 1), encoding="utf-8")
    problems = set_version.mismatches(tree)
    assert len(problems) == 1
    assert problems[0].startswith("docker-compose.truenas-app.yml:") and "0.0.1" in problems[0]


@pytest.mark.parametrize("bad", ["1.2", "v1.2.3", "latest", "1.2.3-rc1", ""])
def test_only_a_full_release_number_is_accepted(tree, bad):
    with pytest.raises(SystemExit):
        set_version.apply(bad, tree)


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SET_VERSION), *args], capture_output=True, text=True, timeout=60)


def test_the_command_line_check_exits_1_on_a_mismatch_and_0_after_the_fix(tree):
    ok = run_cli("--check", "--root", str(tree))
    assert ok.returncode == 0, ok.stderr
    (tree / ".env.truenas.example").write_text(
        (tree / ".env.truenas.example").read_text(encoding="utf-8").replace(
            f"IMAGE_TAG={release()}", "IMAGE_TAG=0.0.1", 1), encoding="utf-8")
    bad = run_cli("--check", "--root", str(tree))
    assert bad.returncode == 1 and ".env.truenas.example" in bad.stderr and "0.0.1" in bad.stderr
    fixed = run_cli(release(), "--root", str(tree))
    assert fixed.returncode == 0, fixed.stderr
    assert run_cli("--check", "--root", str(tree)).returncode == 0


def test_the_command_line_moves_the_release(tree):
    result = run_cli("9.8.7", "--root", str(tree))
    assert result.returncode == 0, result.stderr
    assert "Release set to 9.8.7" in result.stdout
    assert set_version.source_version(tree) == "9.8.7"
