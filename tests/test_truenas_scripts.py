"""scripts/truenas/lib.sh and pull-images.sh: the shared functions, and the service catalogue.

The catalogue in lib.sh is the one table the three helper scripts read (which images exist, which
port a service listens on, how much VRAM and disk it needs). It is checked here against the two
places that state the same facts, the compose file and the VRAM table of the guide, so a change on
either side fails a test instead of drifting.

The scripts run for real under a fake PATH (tests/truenas_helpers.py): behaviour, not text.
"""

from __future__ import annotations

import re
import shutil
import sys

import pytest

from compose_helpers import REPO_ROOT, load_compose, require_yaml
from truenas_helpers import CORE, SERVICE_IMAGE, FakeHost, combined

if shutil.which("bash") is None or not sys.platform.startswith("linux"):
    pytest.skip("the TrueNAS helper scripts target Linux (GNU coreutils) and need bash", allow_module_level=True)

yaml = require_yaml()

COMPOSE_APP = REPO_ROOT / "docker-compose.truenas-app.yml"
GUIDE = REPO_ROOT / "docs" / "truenas-installation-guide.md"
APP_YAML = REPO_ROOT / "truenas" / "tts-stt" / "app.yaml"


@pytest.fixture
def host(tmp_path):
    return FakeHost(tmp_path)


def bash_value(host, snippet: str) -> str:
    result = host.bash(snippet)
    assert result.returncode == 0, combined(result)
    return result.stdout.strip()


# --- version helpers ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ("0.2.0", "0.2.0", "0"),
        ("0.10.0", "0.9.0", "1"),  # numeric, not lexical
        ("0.9.9", "0.10.0", "-1"),
        ("1.0.0", "0.99.99", "1"),
        ("v0.3.0", "0.2.9", "1"),  # a leading v is ignored
        ("0.3.0-rc1", "0.3.0", "0"),  # so is a pre-release suffix
        ("0.3", "0.3.0", "0"),  # a missing component counts as 0
        ("0.08.0", "0.8.0", "0"),  # leading zeros must not be read as octal
    ],
)
def test_version_compare(host, a, b, expected):
    assert bash_value(host, f'ts_version_cmp "{a}" "{b}"') == expected


@pytest.mark.parametrize(
    "tag, is_release",
    [("0.2.0", True), ("10.20.30", True), ("latest", False), ("0.2", False), ("v0.2.0", False),
     ("0.2.0-nemo2", False), ("sha-1a2b3c4", False), ("master", False)],
)
def test_only_a_plain_release_number_counts_as_a_pinned_release(host, tag, is_release):
    result = host.bash(f'ts_is_semver "{tag}"')
    assert (result.returncode == 0) is is_release


def test_the_highest_release_among_mixed_tags(host):
    """latest, sha and -nemo2 tags must never win, and 0.10.0 beats 0.9.0."""
    tags = ["latest", "0.9.0", "0.10.0", "sha-abc1234", "0.2.0-nemo2", "0.10.0-nemo2", "master", "1.0.0-rc1"]
    out = bash_value(host, "printf '%s\\n' " + " ".join(f"'{t}'" for t in tags) + " | ts_highest_semver")
    assert out == "0.10.0"


def test_no_release_among_the_tags_prints_nothing_and_succeeds(host):
    result = host.bash("printf '%s\\n' latest master sha-abc | ts_highest_semver")
    assert result.returncode == 0
    assert result.stdout.strip() == ""


@pytest.mark.parametrize(
    "ref, registry, name, tag",
    [
        ("ghcr.io/steffenhebestreit/tts-stt-stt:0.2.0", "ghcr.io", "steffenhebestreit/tts-stt-stt", "0.2.0"),
        ("ghcr.io/steffenhebestreit/tts-stt-stt", "ghcr.io", "steffenhebestreit/tts-stt-stt", "latest"),
        ("localhost:5000/team/tts-stt-stt:1.2.3", "localhost:5000", "team/tts-stt-stt", "1.2.3"),
        ("localhost:5000/team/tts-stt-stt", "localhost:5000", "team/tts-stt-stt", "latest"),
        ("ghcr.io/x/tts-stt-stt:0.2.0@sha256:abcdef", "ghcr.io", "x/tts-stt-stt", "0.2.0"),
    ],
)
def test_an_image_reference_is_split_into_registry_name_and_tag(host, ref, registry, name, tag):
    out = bash_value(host, f'ts_split_ref "{ref}"; echo "$TS_REF_REGISTRY|$TS_REF_NAME|$TS_REF_TAG"')
    assert out == f"{registry}|{name}|{tag}"


def test_the_default_release_is_the_one_in_app_yaml(host):
    """The scripts ship in the same checkout as the release they install."""
    declared = yaml.safe_load(APP_YAML.read_text(encoding="utf-8"))["app_version"]
    assert bash_value(host, "ts_default_tag") == declared


def test_image_reference_uses_the_catalogue_suffix(host):
    assert bash_value(host, 'ts_image_ref ghcr.io/me stt-service 0.2.0') == "ghcr.io/me/tts-stt-stt:0.2.0"
    assert bash_value(host, 'ts_image_ref ghcr.io/me whisper-cpp latest') == "ghcr.io/me/tts-stt-whisper-cpp:latest"


def test_decimal_helpers(host):
    assert bash_value(host, "ts_sum 1.6 4 2.5") == "8.1"
    assert host.bash("ts_ge 8.1 8.1").returncode == 0
    assert host.bash("ts_ge 8.0 8.1").returncode != 0


# --- the catalogue against the compose file and the guide ----------------------------------------


def catalogue(host) -> list[dict]:
    rows = []
    for line in bash_value(host, "ts_catalogue").splitlines():
        name, suffix, port, group, profile, vram, disk = line.split("|")
        rows.append({"name": name, "suffix": suffix, "port": int(port), "group": group,
                     "profile": profile, "vram": float(vram), "disk": float(disk)})
    return rows


def compose_services() -> dict:
    return load_compose(COMPOSE_APP)["services"]


def test_every_service_the_compose_file_runs_is_in_the_catalogue(host):
    known = {row["name"] for row in catalogue(host)}
    running = set(compose_services()) - {"diun", "piper-voices-seed"}  # a watcher and a one-shot job
    assert running == known


def test_catalogue_images_ports_and_profiles_match_the_compose_file(host):
    services = compose_services()
    for row in catalogue(host):
        service = services[row["name"]]
        image = service["image"]
        assert f"/tts-stt-{row['suffix']}:" in image, f"{row['name']}: image {image}"

        probe = " ".join(service["healthcheck"]["test"])
        assert f"localhost:{row['port']}/" in probe, f"{row['name']}: listens on {row['port']}?"

        profiles = service.get("profiles") or []
        if row["group"] == "core":
            assert not profiles and not row["profile"], f"{row['name']} is core but has profile {profiles}"
        else:
            assert profiles == [row["profile"]], f"{row['name']}: profile {profiles} vs {row['profile']}"


def test_catalogue_matches_the_images_the_registry_fake_and_ci_publish(host):
    assert {row["name"]: row["suffix"] for row in catalogue(host)} == SERVICE_IMAGE


def test_core_group_is_exactly_the_default_stack(host):
    assert bash_value(host, "ts_services core").split() == CORE


def guide_vram_rows() -> dict[str, list[tuple[str, float, float | None]]]:
    """service -> [(label, vram GB, first-start download GB or None)] in table order."""
    rows: dict[str, list] = {}
    section = GUIDE.read_text(encoding="utf-8").split("## 6. VRAM and disk planning", 1)[1].split("\n## 7.", 1)[0]
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3 or not cells[0].startswith("`"):
            continue
        match = re.match(r"`([a-z0-9-]+)`(.*)", cells[0])
        vram = re.fullmatch(r"([0-9.]+) GB", cells[1])
        if not match or not vram:
            continue
        disk = re.fullmatch(r"([0-9.]+) GB", cells[2])
        rows.setdefault(match.group(1), []).append(
            (match.group(2).strip(), float(vram.group(1)), float(disk.group(1)) if disk else None)
        )
    return rows


def test_catalogue_vram_and_disk_match_the_guide_table(host):
    guide = guide_vram_rows()
    for row in catalogue(host):
        default_label, vram, disk = guide[row["name"]][0]  # the default model is listed first
        assert vram == row["vram"], f"{row['name']} ({default_label}): guide {vram} GB, lib.sh {row['vram']} GB"
        assert disk == row["disk"], f"{row['name']}: guide downloads {disk} GB, lib.sh {row['disk']} GB"


def test_the_non_default_model_sizes_preflight_uses_are_in_the_guide(host):
    """preflight.sh swaps the figure for large-v3 (3.1 GB) and Qwen3-TTS 1.7B (4.5 GB)."""
    guide = guide_vram_rows()
    assert any(v == 3.1 and "large-v3`" in label for label, v, _ in guide["stt-service"])
    assert any(v == 4.5 and "1.7B" in label for label, v, _ in guide["qwen3-tts-service"])


# --- pull-images.sh -------------------------------------------------------------------------------


def test_dry_run_lists_the_default_images_and_touches_no_docker(host):
    host.remove("docker")
    result = host.run("pull-images.sh", "--tag", "0.2.0", "--dry-run")
    assert result.returncode == 0, combined(result)
    pulled = re.findall(r"would pull (\S+)", result.stdout)
    assert pulled == [f"ghcr.io/steffenhebestreit/tts-stt-{SERVICE_IMAGE[s]}:0.2.0" for s in CORE]
    assert host.calls() == []


def test_optional_services_are_added_by_profile_name_or_service_name(host):
    result = host.run("pull-images.sh", "--tag", "0.2.0", "--with", "canary-asr, training,whisper-cpp", "--dry-run")
    assert result.returncode == 0, combined(result)
    pulled = re.findall(r"tts-stt-([a-z0-9-]+):0.2.0", result.stdout)
    assert pulled == [SERVICE_IMAGE[s] for s in CORE] + ["canary-asr", "piper-training", "whisper-cpp"]


def test_only_pulls_just_the_named_services_with_the_registry_override(host):
    result = host.run("pull-images.sh", "--tag", "1.2.3", "--only", "frontend-service,stt-service",
                      "--registry", "registry.example/me", "--dry-run")
    assert re.findall(r"would pull (\S+)", result.stdout) == [
        "registry.example/me/tts-stt-frontend:1.2.3", "registry.example/me/tts-stt-stt:1.2.3"]


def test_without_a_tag_it_uses_the_release_of_this_checkout(host):
    declared = yaml.safe_load(APP_YAML.read_text(encoding="utf-8"))["app_version"]
    result = host.run("pull-images.sh", "--dry-run")
    assert result.returncode == 0, combined(result)
    assert f"tts-stt-frontend:{declared}" in result.stdout


@pytest.mark.parametrize("args", [["--with", "nonsense"], ["--only", "nonsense"], ["--frobnicate"]])
def test_bad_arguments_exit_2(host, args):
    result = host.run("pull-images.sh", "--tag", "0.2.0", "--dry-run", *args)
    assert result.returncode == 2, combined(result)


def test_it_pulls_each_image_and_only_pulls(host):
    result = host.run("pull-images.sh", "--tag", "0.2.0")
    assert result.returncode == 0, combined(result)
    pulls = [c[-1] for c in host.calls() if c[0] == "docker" and c[1] == "pull"]
    assert pulls == [f"ghcr.io/steffenhebestreit/tts-stt-{SERVICE_IMAGE[s]}:0.2.0" for s in CORE]
    assert host.docker_subcommands() <= {"info", "pull"}, "pull-images.sh must not touch the app"


def test_a_failed_pull_exits_1_names_the_image_and_points_at_update_check(host):
    bad = "ghcr.io/steffenhebestreit/tts-stt-qwen3-tts:9.9.9"
    host.state["docker"]["pull_rc"][bad] = 1
    result = host.run("pull-images.sh", "--tag", "9.9.9")
    out = combined(result)
    assert result.returncode == 1
    assert bad in out.split("These pulls failed:")[1]
    assert "update-check.sh --tag 9.9.9" in out
    # the others were still attempted: one failure does not hide the rest
    assert len([c for c in host.calls() if c[:2] == ["docker", "pull"]]) == len(CORE)


def test_without_a_usable_docker_it_says_so(host):
    host.state["docker"]["usable"] = False
    result = host.run("pull-images.sh", "--tag", "0.2.0")
    assert result.returncode == 1
    assert "cannot talk to Docker" in combined(result)
