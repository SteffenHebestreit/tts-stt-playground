"""The publishing policy: which images a trigger builds and which tags it may move.

publish-images.yml used to publish ``latest`` from any branch on a manual run, let
one failing CUDA build block every image, and gave nothing a test could call. The
policy now lives in scripts/plan_publish.py and the workflow carries it out; these
tests call the planner the way the workflow does, check that its image list agrees
with the compose files that consume the images, and run the two shell steps of the
manifest job (the digest check and the manifest assembly) against a stub ``docker``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
from compose_helpers import REPO_ROOT, load_compose, yaml

_spec = spec_from_file_location("plan_publish", REPO_ROOT / "scripts" / "plan_publish.py")
plan_publish = module_from_spec(_spec)
sys.modules["plan_publish"] = plan_publish
_spec.loader.exec_module(plan_publish)  # type: ignore[union-attr]

PlanError = plan_publish.PlanError
make_plan = plan_publish.make_plan

WORKFLOW = REPO_ROOT / ".github" / "workflows" / "publish-images.yml"
CI = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _legs(plan, key: str) -> list[dict]:
    return [leg for leg in plan.build if leg["key"] == key]


# --- what each trigger may publish -----------------------------------------------------


def test_a_release_tag_publishes_versions_and_moves_latest():
    plan = make_plan("push", "refs/tags/v0.2.0")
    assert (plan.release, plan.latest, plan.nemo2, plan.extra_tag) == (True, True, True, "")


def test_a_prerelease_tag_is_published_but_never_moves_latest():
    plan = make_plan("push", "refs/tags/v1.10.3-rc.1")
    assert (plan.release, plan.latest) == (True, False)


def test_master_moves_latest_but_writes_no_version_tag_and_skips_the_rollback_images():
    plan = make_plan("push", "refs/heads/master")
    assert (plan.release, plan.latest, plan.nemo2) == (False, True, False)
    assert not [leg for leg in plan.build if leg["suffix"]]


@pytest.mark.parametrize("ref", ["refs/heads/develop", "refs/heads/feature/x", "refs/pull/3/merge"])
def test_a_push_to_any_other_ref_is_refused(ref):
    with pytest.raises(PlanError, match="not published"):
        make_plan("push", ref)


@pytest.mark.parametrize("tag", ["v1", "v1.2", "vfoo", "v1.2.3x", "1.2.3", "v1.2.3-", "v1.2.3+build"])
def test_a_tag_that_is_not_a_release_is_refused_instead_of_published_oddly(tag):
    with pytest.raises(PlanError, match="not a release tag"):
        make_plan("push", f"refs/tags/{tag}")


@pytest.mark.parametrize("ref", [
    "refs/heads/master", "refs/heads/develop", "refs/heads/feature/x", "refs/tags/v0.2.0"])
def test_a_manual_run_never_moves_latest_wherever_it_starts(ref):
    """The defect: workflow_dispatch defaulted to `latest` from any branch."""
    plan = make_plan("workflow_dispatch", ref)
    assert plan.latest is False and plan.release is False


@pytest.mark.parametrize("tag", ["latest", "LATEST", " latest "])
def test_a_manual_run_cannot_ask_for_latest(tag):
    with pytest.raises(PlanError, match="never publishes 'latest'"):
        make_plan("workflow_dispatch", "refs/heads/develop", extra_tag=tag)


@pytest.mark.parametrize("tag", ["0.2.0", "v0.2.0", "1.2", "v1.2", "2.0.0-rc.1"])
def test_a_manual_run_cannot_overwrite_a_release_tag(tag):
    """A TrueNAS install pins X.Y.Z; only a pushed vX.Y.Z tag may write it."""
    with pytest.raises(PlanError, match="release version"):
        make_plan("workflow_dispatch", "refs/heads/develop", extra_tag=tag)


@pytest.mark.parametrize("tag", ["a b", "../x", "-lead", ".lead", "x" * 129, "ü", "1.2.3+b"])
def test_a_manual_run_refuses_what_is_not_a_docker_tag(tag):
    with pytest.raises(PlanError, match="not a valid Docker tag"):
        make_plan("workflow_dispatch", "refs/heads/develop", extra_tag=tag)


def test_a_manual_run_accepts_a_plain_extra_tag_and_the_rollback_switch():
    plan = make_plan("workflow_dispatch", "refs/heads/develop", extra_tag="my-test_1", nemo2=True)
    assert (plan.extra_tag, plan.nemo2) == ("my-test_1", True)
    assert make_plan("workflow_dispatch", "refs/heads/develop", extra_tag="").extra_tag == ""


def test_an_extra_tag_on_a_push_is_refused():
    with pytest.raises(PlanError):
        make_plan("push", "refs/heads/master", extra_tag="x")


def test_other_events_do_not_publish():
    with pytest.raises(PlanError, match="does not publish"):
        make_plan("pull_request", "refs/pull/1/merge")


# --- the matrices ----------------------------------------------------------------------------


def test_every_manifest_job_has_exactly_the_builds_it_will_look_for():
    """`digests-<key>_*` must match this image's artifacts and nobody else's, or the
    parakeet manifest would pick up the digest of its own -nemo2 rollback image."""
    import fnmatch

    plan = make_plan("push", "refs/tags/v0.2.0")
    artifacts = [leg["artifact"] for leg in plan.build]
    assert len(set(artifacts)) == len(artifacts), "two builds upload the same artifact name"
    for merge in plan.merge:
        found = sorted(a for a in artifacts if fnmatch.fnmatchcase(a, f"digests-{merge['key']}_*"))
        expected = sorted(f"digests-{merge['key']}_{arch}" for arch in merge["archs"].split())
        assert found == expected, merge


def test_multi_arch_images_build_on_native_runners_and_the_rest_are_amd64_only():
    plan = make_plan("push", "refs/heads/master")
    for leg in plan.build:
        assert leg["runner"] == ("ubuntu-24.04-arm" if leg["arch"] == "arm64" else "ubuntu-latest")
    arm = {leg["name"] for leg in plan.build if leg["arch"] == "arm64"}
    assert arm == {"frontend", "piper-tts", "whisper-cpp"}


def test_the_rollback_images_are_built_against_nemo_2_and_leave_the_cache_alone():
    plan = make_plan("push", "refs/tags/v0.2.0")
    rollback = [leg for leg in plan.build if leg["suffix"]]
    assert {leg["name"] for leg in rollback} == {"parakeet-asr", "canary-asr"}
    for leg in rollback:
        assert leg["build_args"] == "NEMO_TOOLKIT_SPEC=>=2.7.3,<3"
        assert leg["suffix"] == "-nemo2" and leg["key"].endswith("-nemo2")
        assert leg["cache_to"] == "", "the rollback build must not export into the main image's cache scope"
        assert leg["cache_from"].startswith("type=gha,scope=")
    main = [leg for leg in plan.build if not leg["suffix"]]
    assert all(leg["build_args"] == "" for leg in main), (
        "the pins live in the Dockerfiles; passing one here would make a published image "
        "differ from `docker compose build`")


def test_the_matrices_survive_the_github_output_round_trip():
    plan = make_plan("push", "refs/tags/v0.2.0")
    out = plan.outputs()
    assert json.loads(out["build"])["include"] == plan.build
    assert json.loads(out["merge"])["include"] == plan.merge
    assert (out["release"], out["latest"], out["nemo2"]) == ("true", "true", "true")
    assert "\n" not in out["build"] and "\n" not in out["merge"]


# --- the planner agrees with the files that consume the images -------------------------------------


def _compose_images() -> dict[str, dict]:
    """image name (after tts-stt-) -> {service, context, dockerfile} from docker-compose.yml."""
    services = load_compose(REPO_ROOT / "docker-compose.yml")["services"]
    found = {}
    for name, service in services.items():
        if name == "piper-voices-seed":
            continue  # a one-shot job that runs on the piper-tts image
        image = service.get("image", "")
        match = re.search(r"/tts-stt-([a-z0-9-]+):", image)
        if match and isinstance(service.get("build"), dict):
            found.setdefault(match.group(1), {"service": name, **service["build"], "image": image})
    return found


def test_the_planner_publishes_exactly_the_images_compose_runs():
    published = {name for name, _, _ in plan_publish.SERVICES}
    assert published == set(_compose_images()), (
        "compose runs images this workflow never publishes (a deployment would pull "
        "nothing), or publishes images nothing uses")


def test_each_published_context_is_the_one_compose_builds_and_has_a_dockerfile():
    compose = _compose_images()
    for name, context, _ in plan_publish.SERVICES:
        assert compose[name]["context"].rstrip("/") == context.rstrip("/"), name
        assert (REPO_ROOT / context / "Dockerfile").is_file(), f"{context}/Dockerfile"


def test_the_arm64_images_are_the_services_the_arm64_overlay_supports():
    overlay = load_compose(REPO_ROOT / "docker-compose.arm64.yml")["services"]
    compose = _compose_images()
    arm = {name for name, _, archs in plan_publish.SERVICES if "arm64" in archs}
    assert {compose[name]["service"] for name in arm} == set(overlay)


def test_rollback_images_exist_only_for_services_compose_can_switch_and_whose_dockerfile_takes_the_spec():
    compose = _compose_images()
    for name in plan_publish.NEMO2_SERVICES:
        assert "NEMO_IMAGE_SUFFIX" in compose[name]["image"], (
            f"{name}: compose cannot select the -nemo2 image this workflow publishes")
        dockerfile = (REPO_ROOT / dict((n, c) for n, c, _ in plan_publish.SERVICES)[name] / "Dockerfile")
        assert re.search(r"^ARG\s+NEMO_TOOLKIT_SPEC=", dockerfile.read_text(encoding="utf-8"), re.M), (
            f"{dockerfile}: the rollback build argument is not declared, so it would be ignored")
    switchable = {n for n, s in compose.items() if "NEMO_IMAGE_SUFFIX" in s["image"]}
    assert switchable == set(plan_publish.NEMO2_SERVICES)
    assert plan_publish.NEMO2_SUFFIX == "-nemo2"


def test_the_image_prefix_is_lower_case():
    """GHCR and BuildKit reject capitals; github.repository_owner keeps the account's own."""
    assert plan_publish.image_prefix("SteffenHebestreit") == "ghcr.io/steffenhebestreit/tts-stt-"
    with pytest.raises(PlanError):
        plan_publish.image_prefix("")


def _run_cli(env: dict, tmp_path: Path, *argv: str):
    out = tmp_path / "out"
    out.write_text("")
    process = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "plan_publish.py"), *argv],
        env={**os.environ, "GITHUB_OUTPUT": str(out), **env}, capture_output=True, text=True, cwd=tmp_path)
    return process, out.read_text()


def test_the_command_line_writes_github_outputs_from_the_environment(tmp_path):
    process, written = _run_cli({
        "GITHUB_EVENT_NAME": "workflow_dispatch", "GITHUB_REF": "refs/heads/develop",
        "GITHUB_REPOSITORY_OWNER": "SteffenHebestreit", "PUBLISH_EXTRA_TAG": "try-1",
        "PUBLISH_NEMO2": "true"}, tmp_path)
    assert process.returncode == 0, process.stderr
    values = dict(line.split("=", 1) for line in written.splitlines())
    assert values["latest"] == "false" and values["nemo2"] == "true"
    assert values["extra_tag"] == "try-1"
    assert values["image_prefix"] == "ghcr.io/steffenhebestreit/tts-stt-"
    assert json.loads(values["merge"])["include"]


def test_the_command_line_fails_the_run_with_the_reason(tmp_path):
    process, written = _run_cli({
        "GITHUB_EVENT_NAME": "workflow_dispatch", "GITHUB_REF": "refs/heads/master",
        "GITHUB_REPOSITORY_OWNER": "x", "PUBLISH_EXTRA_TAG": "latest", "GITHUB_ACTIONS": "true"}, tmp_path)
    assert process.returncode == 1
    assert "::error" in process.stderr and "latest" in process.stderr
    assert written == "", "a refused plan must not leave outputs a later job could act on"


# --- the workflow carries the plan out -----------------------------------------------------------------


def test_the_builds_wait_for_the_ci_checks_and_the_checks_are_the_reusable_ci_workflow():
    jobs = _workflow()["jobs"]
    assert jobs["tests"]["uses"] == "./.github/workflows/ci.yml"
    assert {"plan", "tests"} <= set(jobs["build"]["needs"])
    inputs = yaml.safe_load(CI.read_text(encoding="utf-8"))[True]["workflow_call"]["inputs"]
    assert set(jobs["tests"]["with"]) <= set(inputs), "the call passes an input ci.yml does not declare"
    assert jobs["tests"]["with"]["image-builds"] is False, "publish builds every image itself"


def test_one_failing_image_does_not_block_the_others():
    """The defect: `needs: [build]` with no condition skipped every manifest job."""
    merge = _workflow()["jobs"]["merge"]
    condition = str(merge["if"])
    assert "!cancelled()" in condition, "a failed build leg would skip every manifest job"
    assert "needs.tests.result == 'success'" in condition, "nothing may be published when CI failed"
    assert "needs.build.result" not in condition, "gating on the whole build job blocks every image"
    assert merge["strategy"]["fail-fast"] is False
    assert "build" in merge["needs"]


def test_only_the_jobs_that_push_get_write_access_to_packages():
    workflow = _workflow()
    assert "packages" not in (workflow.get("permissions") or {}), "least privilege at the top level"
    for name, job in workflow["jobs"].items():
        wants = (job.get("permissions") or {}).get("packages") == "write"
        assert wants == (name in {"build", "merge"}), name


def test_the_workflow_dispatch_inputs_match_what_the_planner_reads():
    workflow = _workflow()
    inputs = workflow[True]["workflow_dispatch"]["inputs"]
    assert set(inputs) == {"extra-tag", "nemo2"}
    env = workflow["jobs"]["plan"]["steps"][-1]["env"]
    assert set(env) == {"PUBLISH_EXTRA_TAG", "PUBLISH_NEMO2"}
    assert "inputs.extra-tag" in env["PUBLISH_EXTRA_TAG"] and "inputs.nemo2" in env["PUBLISH_NEMO2"]
    assert workflow[True]["push"]["branches"] == ["master"]


# The two shell steps of the manifest job, run for real against a stub `docker`.

needs_shell = pytest.mark.skipif(
    not (shutil.which("bash") and shutil.which("jq")), reason="bash and jq are needed to run the steps")


def _step(name: str) -> dict:
    return next(s for s in _workflow()["jobs"]["merge"]["steps"] if s.get("name") == name)


def _bash(step: dict, tmp_path: Path, env: dict, cwd: Path):
    script = tmp_path / "step.sh"
    script.write_text("set -eo pipefail\n" + step["run"], encoding="utf-8")
    return subprocess.run(["bash", str(script)], env=env, cwd=cwd, capture_output=True, text=True)


@needs_shell
def test_the_digest_check_names_the_architecture_that_is_missing(tmp_path):
    step = _step("Check every architecture of this image was built")
    # the step reads /tmp/digests: point it at the test directory
    step = {**step, "run": step["run"].replace("/tmp/digests", str(tmp_path / "digests"))}
    root = tmp_path / "digests"
    (root / "digests-frontend_amd64").mkdir(parents=True)
    (root / "digests-frontend_amd64" / "abc123").write_text("")
    env = {**os.environ, "KEY": "frontend", "EXPECTED": "amd64 arm64"}

    partial = _bash(step, tmp_path, env, tmp_path)
    assert partial.returncode == 1
    assert "arm64" in partial.stdout and "frontend not published" in partial.stdout
    assert "amd64" not in partial.stdout.split("no image was built for:")[1].split(".")[0]

    (root / "digests-frontend_arm64").mkdir()
    (root / "digests-frontend_arm64" / "def456").write_text("")
    assert _bash(step, tmp_path, env, tmp_path).returncode == 0

    # an image whose only architecture never built (no artifact at all) also fails
    lonely = {**env, "KEY": "stt", "EXPECTED": "amd64"}
    assert _bash(step, tmp_path, lonely, tmp_path).returncode == 1


@needs_shell
def test_the_manifest_is_assembled_from_the_exported_json_and_this_images_digests(tmp_path):
    step = _step("Create and push manifest list")
    digests = tmp_path / "digests"
    for arch, digest in (("amd64", "aaa111"), ("arm64", "bbb222")):
        (digests / f"digests-frontend_{arch}").mkdir(parents=True)
        (digests / f"digests-frontend_{arch}" / digest).write_text("")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "docker.log"
    stub = bin_dir / "docker"
    stub.write_text(f'#!/bin/sh\nfor a in "$@"; do printf "%s\\n" "$a" >> "{log}"; done\n')
    stub.chmod(0o755)
    hostile = "0.2.0'; touch pwned; echo '"
    metadata = json.dumps({"tags": [
        "ghcr.io/o/tts-stt-frontend:0.2.0", "ghcr.io/o/tts-stt-frontend:latest", hostile]})
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
           "IMAGE": "ghcr.io/o/tts-stt-frontend", "DOCKER_METADATA_OUTPUT_JSON": metadata}
    work = tmp_path / "work"
    work.mkdir()
    step = {**step, "run": f'cd "{digests}"\n' + step["run"]}

    result = _bash(step, tmp_path, env, work)

    assert result.returncode == 0, result.stderr
    args = log.read_text().splitlines()
    assert args[:3] == ["buildx", "imagetools", "create"]
    tags = [args[i + 1] for i, a in enumerate(args) if a == "-t"]
    assert tags == ["ghcr.io/o/tts-stt-frontend:0.2.0", "ghcr.io/o/tts-stt-frontend:latest", hostile]
    refs = [a for a in args if "@sha256:" in a]
    assert refs == ["ghcr.io/o/tts-stt-frontend@sha256:aaa111", "ghcr.io/o/tts-stt-frontend@sha256:bbb222"]
    assert not (work / "pwned").exists() and not (digests / "pwned").exists(), (
        "the tag list reached the shell as code")
