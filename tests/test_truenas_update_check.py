"""scripts/truenas/update-check.sh against a fake host and a fake registry.

The property that matters most is that it is READ-ONLY: it may inspect containers and ask a
registry, and must never pull, run, stop or redeploy anything. The rest is the verdict: is a newer
release published, is the image behind a floating tag newer, and does it degrade to "could not
check" (never to "up to date") when the registry does not answer.

Registry access has three routes (docker buildx, skopeo, plain curl). Each is exercised alone by
removing the other tools from the fake PATH.
"""

from __future__ import annotations

import shutil
import sys

import pytest

from truenas_helpers import CORE, SERVICE_IMAGE, FakeHost, combined

if shutil.which("bash") is None or not sys.platform.startswith("linux"):
    pytest.skip("the TrueNAS helper scripts target Linux (GNU coreutils) and need bash", allow_module_level=True)

READ_ONLY_DOCKER = {"info", "ps", "inspect", "image inspect", "buildx imagetools inspect"}


@pytest.fixture
def host(tmp_path):
    return FakeHost(tmp_path)


def check(host, *args, env=None):
    result = host.run("update-check.sh", *args, env=env)
    return result, combined(result)


def only_route(host, route: str) -> None:
    """Leave exactly one way to reach the registry."""
    host.state["methods"] = {"buildx": route == "buildx", "skopeo": route == "skopeo", "curl": route == "curl"}
    for tool, name in (("skopeo", "skopeo"), ("curl", "curl")):
        if route != name:
            host.remove(tool)


ROUTES = ["buildx", "skopeo", "curl"]


# --- has this release been published? -----------------------------------------------------------------


@pytest.mark.parametrize("route", ROUTES)
def test_a_published_release_is_reported_published(host, route):
    host.publish("0.3.0")
    only_route(host, route)
    result, out = check(host, "--tag", "0.3.0")
    assert result.returncode == 0, out
    assert out.count("published  ghcr.io/steffenhebestreit/tts-stt-") == len(CORE)
    assert "Tag 0.3.0 is published for all 5 checked images." in out


def test_a_release_that_does_not_exist_yet_is_missing_and_exits_10(host):
    host.publish("0.2.0")
    result, out = check(host, "--tag", "0.3.0")
    assert result.returncode == 10, out
    assert out.count("MISSING") == len(CORE)
    assert "do not deploy it yet" in out


def test_a_half_published_release_is_not_deployable(host):
    """CI publishes the images in parallel: one lagging image must hold everyone back."""
    host.publish("0.2.0")
    host.publish("0.3.0", services=[s for s in CORE if s != "qwen3-tts-service"])
    result, out = check(host, "--tag", "0.3.0")
    assert result.returncode == 10, out
    missing = [line for line in out.splitlines() if "MISSING" in line]
    assert len(missing) == 1 and "tts-stt-qwen3-tts:0.3.0" in missing[0]


def test_an_unreachable_registry_is_unknown_never_published_or_missing(host):
    host.state["reachable"] = False
    result, out = check(host, "--tag", "0.3.0")
    assert result.returncode == 2, out
    assert "MISSING" not in out and out.count("UNKNOWN") == len(CORE)
    assert "Could not verify tag 0.3.0" in out


def test_a_private_or_unknown_package_is_unknown(host):
    """No repository at all (private package): the tag list fails too, so it is not called missing."""
    result, out = check(host, "--tag", "0.3.0")
    assert result.returncode == 2 and "UNKNOWN" in out


def test_optional_services_can_be_checked_with_the_release(host):
    host.publish("0.2.0")  # the optional images exist, only not at 0.3.0
    host.publish("0.3.0", services=CORE)
    result, out = check(host, "--tag", "0.3.0", "--with", "canary-asr,training")
    assert result.returncode == 10, out
    assert "tts-stt-canary-asr:0.3.0" in out and "tts-stt-piper-training:0.3.0" in out


def test_an_unknown_optional_service_is_a_usage_error(host):
    result, out = check(host, "--tag", "0.3.0", "--with", "nonsense")
    assert result.returncode == 2 and "unknown optional service" in out


def test_a_fork_registry_is_honoured(host):
    host.publish("0.3.0", registry="registry.example/me")
    result, out = check(host, "--tag", "0.3.0", "--registry", "registry.example/me")
    assert result.returncode == 0, out


def test_checking_a_tag_needs_no_running_app(host):
    host.publish("0.3.0")
    result, _ = check(host, "--tag", "0.3.0")
    assert result.returncode == 0
    assert "ps" not in host.docker_subcommands()


# --- the running app ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("route", ["skopeo", "curl"])
def test_an_app_running_the_published_images_is_up_to_date(host, route):
    host.publish("0.2.0")
    host.run_app("0.2.0")
    only_route(host, route)
    result, out = check(host)
    assert result.returncode == 0, out
    assert "Up to date." in out and "UPDATE" not in out
    assert out.count("up to date") == len(CORE)


def test_with_only_docker_buildx_the_digests_compare_but_new_releases_cannot_be_detected(host):
    """buildx reads a digest and cannot list tags: say so instead of claiming 'up to date'."""
    host.publish("0.2.0")
    host.publish("0.3.0")
    host.run_app("0.2.0")
    only_route(host, "buildx")
    result, out = check(host)
    assert result.returncode == 2, out
    assert out.count("up to date (") == len(CORE), "the digest comparison itself works"
    assert "Could not list release tags" in out and "Up to date." not in out


def test_a_floating_tag_whose_image_moved_is_an_update(host):
    host.publish("latest")
    host.run_app("latest", stale=["stt-service"])
    result, out = check(host)
    assert result.returncode == 10, out
    stt = [line for line in out.splitlines() if line.startswith("stt-service")][0]
    assert "UPDATE" in stt and "registry has sha256:" in stt
    assert "the registry holds a newer image for a tag you follow" in out
    assert "./pull-images.sh --tag latest --only stt-service\n" in out, "names the moved service, not all of them"
    assert "Stop, then Start" in out and "PULL_POLICY set to always" in out
    assert "UPDATE AVAILABLE: release" not in out, "a floating tag has no release number to announce"


@pytest.mark.parametrize("route", ["skopeo", "curl"])
def test_a_newer_release_is_announced_once_every_image_of_it_exists(host, route):
    """Listing tags needs skopeo or curl: docker buildx can only read a digest."""
    host.publish("0.2.0")
    host.publish("0.3.0")
    host.run_app("0.2.0")
    only_route(host, route)
    result, out = check(host)
    assert result.returncode == 10, out
    assert "UPDATE AVAILABLE: release 0.3.0 is published (you run 0.2.0)." in out
    assert "pull-images.sh --tag 0.3.0" in out
    assert "docs/truenas-installation-guide.md" in out


def test_buildx_for_digests_and_curl_for_the_tag_list_work_together(host):
    """The common TrueNAS host: docker buildx present, skopeo absent."""
    host.publish("0.2.0")
    host.publish("0.3.0")
    host.run_app("0.2.0")
    host.remove("skopeo")
    host.state["methods"]["skopeo"] = False
    result, out = check(host)
    assert result.returncode == 10 and "release 0.3.0 is published" in out
    assert "buildx imagetools inspect" in host.docker_subcommands()


def test_the_pull_hint_covers_every_service_that_moved_and_only_those(host):
    host.publish("latest")
    host.run_app("latest", stale=["stt-service", "frontend-service"])
    _, out = check(host)
    assert "./pull-images.sh --tag latest --only frontend-service,stt-service\n" in out


def test_the_pull_hint_for_a_newer_release_includes_the_optional_services_that_run(host):
    """pull-images.sh pulls only the default stack unless told: a hint that forgot canary would leave it behind."""
    optional = ["canary-asr-service", "piper-training-service"]
    host.publish("0.2.0")
    host.publish("0.3.0")
    host.run_app("0.2.0", services=CORE + optional)
    result, out = check(host)
    assert result.returncode == 10, out
    assert "./pull-images.sh --tag 0.3.0 --with canary-asr,training\n" in out


def test_the_pull_hint_for_a_newer_release_has_no_with_when_only_the_default_stack_runs(host):
    host.publish("0.2.0")
    host.publish("0.3.0")
    host.run_app("0.2.0")
    _, out = check(host)
    assert "./pull-images.sh --tag 0.3.0\n" in out


def test_the_highest_release_wins_not_the_newest_string(host):
    host.publish("0.2.0")
    host.publish("0.9.0")
    host.publish("0.10.0")
    host.publish("latest")
    host.publish("sha-abc1234")
    host.publish("0.10.0-nemo2")
    host.run_app("0.9.0")
    result, out = check(host)
    assert result.returncode == 10 and "release 0.10.0 is published" in out


def test_a_release_still_being_published_is_not_announced_yet(host):
    """The frontend has 0.3.0 but qwen3-tts does not: updating now would fail on the lagging image."""
    host.publish("0.2.0")
    host.publish("0.3.0", services=["frontend-service"])
    host.run_app("0.2.0")
    result, out = check(host)
    assert "UPDATE AVAILABLE" not in out
    assert "still publishing" in out and "try again in a few minutes" in out
    assert result.returncode == 0, "nothing the user can act on yet"
    assert "Up to date." not in out, "must not claim 'up to date' while a release is on its way"


def test_a_pinned_release_at_the_newest_version_is_up_to_date(host):
    host.publish("0.2.0")
    host.publish("0.1.0")
    host.run_app("0.2.0")
    result, out = check(host)
    assert result.returncode == 0 and "Up to date." in out


def test_a_pinned_release_whose_own_image_was_rebuilt_is_an_update(host):
    """A vX.Y.Z tag republished (a re-run of the release job) changes the digest under the same tag."""
    host.publish("0.2.0")
    host.run_app("0.2.0", stale=["frontend-service"])
    result, out = check(host)
    assert result.returncode == 10, out
    assert "UPDATE" in out


def test_mixed_release_tags_are_called_out(host):
    host.publish("0.2.0")
    host.publish("0.1.0")
    host.run_app("0.2.0", services=CORE[:3])
    host.run_app("0.1.0", services=CORE[3:])
    _, out = check(host)
    assert "run different release tags" in out


def test_other_containers_of_the_project_are_ignored(host):
    """The optional diun container lives in the same project but is not one of our images."""
    host.publish("0.2.0")
    host.run_app("0.2.0")
    host.state["docker"]["containers"].append(
        {"name": "ix-tts-stt-diun-1", "project": "ix-tts-stt", "service": "diun",
         "image": "ghcr.io/crazy-max/diun:latest", "image_id": "sha256:d1", "repo_digests": [], "ports": ""})
    result, out = check(host)
    assert result.returncode == 0, out
    assert "diun" not in out


def test_another_app_name_maps_to_its_compose_project(host):
    host.publish("0.2.0")
    host.run_app("0.2.0", project="ix-voice")
    assert check(host)[0].returncode == 1, "the default app name finds nothing"
    result, out = check(host, "--app", "voice")
    assert result.returncode == 0, out
    result, out = check(host, "--project", "ix-voice")
    assert result.returncode == 0, out


# --- degrading gracefully ---------------------------------------------------------------------------------------


def test_an_unreachable_registry_is_could_not_check_never_up_to_date(host):
    host.publish("0.2.0")
    host.run_app("0.2.0")
    host.state["reachable"] = False
    result, out = check(host)
    assert result.returncode == 2, out
    assert "Up to date." not in out
    assert "Could not check everything" in out and "not a confirmation" in out


def test_an_image_without_a_registry_digest_cannot_be_compared(host):
    """Built or loaded locally: no RepoDigests to compare with."""
    host.publish("0.2.0")
    host.run_app("0.2.0")
    for container in host.state["docker"]["containers"]:
        container["repo_digests"] = []
    result, out = check(host)
    assert result.returncode == 2, out
    assert "no registry digest on the local image" in out and "Up to date." not in out


def test_offline_lists_the_running_images_and_contacts_no_registry(host):
    host.publish("0.2.0")
    host.run_app("0.2.0")
    result, out = check(host, "--offline")
    assert result.returncode == 0, out
    assert "Offline: nothing was compared with a registry." in out
    assert out.count("not checked") == len(CORE)
    assert not [c for c in host.calls() if c[0] in ("curl", "skopeo")]
    assert "buildx imagetools inspect" not in host.docker_subcommands()


def test_quiet_prints_nothing_when_all_is_well_and_exits_0(host):
    host.publish("0.2.0")
    host.run_app("0.2.0")
    result, out = check(host, "--quiet")
    assert result.returncode == 0 and out.strip() == ""


def test_quiet_still_reports_an_update(host):
    host.publish("0.2.0")
    host.publish("0.3.0")
    host.run_app("0.2.0")
    result, out = check(host, "--quiet")
    assert result.returncode == 10 and "UPDATE AVAILABLE: release 0.3.0" in out
    assert "up to date" not in out.lower().replace("update available", ""), "quiet hides the healthy rows"


def test_no_running_containers_is_an_error_that_names_the_project(host):
    result, out = check(host)
    assert result.returncode == 1 and "ix-tts-stt" in out


def test_a_project_without_our_images_is_reported(host):
    host.state["docker"]["containers"].append(
        {"name": "ix-tts-stt-x-1", "project": "ix-tts-stt", "service": "x", "image": "nginx:1.27",
         "image_id": "sha256:1", "repo_digests": [], "ports": ""})
    result, out = check(host)
    assert result.returncode == 1 and "Nothing to check" in out


def test_a_docker_daemon_that_is_down_is_an_error(host):
    host.state["docker"]["usable"] = False
    result, out = check(host)
    assert result.returncode == 1 and "cannot talk to Docker" in out


def test_a_missing_docker_is_an_error(host):
    host.remove("docker")
    result, out = check(host)
    assert result.returncode == 1 and "cannot talk to Docker" in out


def test_bad_arguments_exit_2(host):
    assert check(host, "--frobnicate")[0].returncode == 2


# --- the guarantee ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args, stale",
    [
        ((), []),
        ((), ["stt-service"]),
        (("--offline",), []),
        (("--quiet",), []),
        (("--tag", "0.3.0"), []),
        (("--tag", "0.3.0", "--with", "canary-asr"), []),
    ],
)
def test_update_check_never_changes_anything(host, args, stale):
    """Only reads: no pull, run, compose, stop, rm, restart, and it writes nothing to disk."""
    host.publish("0.2.0")
    host.publish("0.3.0", services=CORE)
    host.run_app("0.2.0", stale=stale)
    before = sorted(p.name for p in host.data_dir.rglob("*"))
    check(host, *args)
    assert host.docker_subcommands() <= READ_ONLY_DOCKER, host.docker_subcommands()
    assert sorted(p.name for p in host.data_dir.rglob("*")) == before


def test_the_image_catalogue_the_fake_registry_uses_is_the_one_the_script_checks(host):
    """Guards the fake itself: a service name the script knows and the fake does not would pass vacuously."""
    host.publish("0.2.0")
    result, out = check(host, "--tag", "0.2.0", "--with",
                         "canary-asr,parakeet-asr,chatterbox-tts,magpie-tts,training,whisper-cpp")
    assert result.returncode == 0, out
    assert out.count("published  ") == len(SERVICE_IMAGE)
