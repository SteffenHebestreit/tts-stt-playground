"""scripts/truenas/preflight.sh against a fake host: every verdict it can give.

The host is described in data (tests/truenas_helpers.py): which tools exist, which GPU and driver
the fake nvidia-smi reports, how much disk is free. The script itself runs unmodified.

What this proves is the script's logic. What it cannot prove is that the real nvidia-smi, docker
and ZFS report what the stubs say; the guide lists what to check on a real system.
"""

from __future__ import annotations

import shutil
import sys
import socket

import pytest

from compose_helpers import REPO_ROOT, load_compose
from truenas_helpers import FakeHost, combined

if shutil.which("bash") is None or not sys.platform.startswith("linux"):
    pytest.skip("the TrueNAS helper scripts target Linux (GNU coreutils) and need bash", allow_module_level=True)


@pytest.fixture
def host(tmp_path):
    return FakeHost(tmp_path)


def preflight(host, *args, env=None):
    result = host.run("preflight.sh", "--data-dir", str(host.data_dir), *args, env=env)
    return result, combined(result)


def gpu(host, **changes):
    host.state["nvidia"]["gpus"][0].update(changes)


def lines_with(text: str, tag: str) -> list[str]:
    return [line for line in text.splitlines() if tag in line]


# --- the happy path and what it must not do ---------------------------------------------------------


def test_a_ready_host_gets_a_ready_verdict(host):
    result, out = preflight(host)
    assert result.returncode == 0, out
    assert "READY" in out and "NOT READY" not in out
    assert "GPU 0 will be used" in out and "Docker has the nvidia runtime" in out
    assert "8.1 GB resident" in out, "the default set (1.6 + 4 + 2.5 GB) is what it budgets"


def test_preflight_is_read_only_towards_docker(host):
    preflight(host)
    assert host.docker_subcommands() <= {"info", "version", "compose version", "ps"}, host.docker_subcommands()


def test_without_create_it_makes_no_directories(host):
    preflight(host)
    assert list(host.data_dir.iterdir()) == []


def test_create_makes_the_subdirectories_the_compose_file_mounts(host):
    result, out = preflight(host, "--create")
    assert result.returncode == 0, out
    assert sorted(p.name for p in host.data_dir.iterdir()) == [
        "backend-settings", "cache", "models", "output", "qwen3-voices", "settings"]
    _, again = preflight(host, "--create")
    assert "all data subdirectories exist" in again


def test_create_makes_the_settings_folders_without_a_gpu_too(host):
    """The web UI's Settings page saves into settings/ on every install, GPU or not.

    On a free port, so that a web UI already running on this machine's port 3000 cannot decide it.
    """
    host.remove("nvidia-smi")
    sock, port = listening_socket()
    sock.close()
    result, out = preflight(host, "--create", "--no-gpu", "--port", str(port))
    assert result.returncode == 0, out
    assert {"settings", "backend-settings"} <= {p.name for p in host.data_dir.iterdir()}
    assert not list((host.data_dir / "settings").iterdir()), "preflight writes no settings, it only makes the folder"


def test_create_adds_the_directories_of_the_optional_services_it_plans_for(host):
    preflight(host, "--create", "--with", "training,whisper-cpp")
    names = {p.name for p in host.data_dir.iterdir()}
    assert {"piper-training-service", "whisper-cpp-models"} <= names
    assert {p.name for p in (host.data_dir / "piper-training-service").iterdir()} == {
        "data", "checkpoints", "models", "configs"}


def test_create_makes_exactly_the_directories_the_compose_file_mounts(host):
    """preflight.sh keeps its own list of subdirectories: it must be the compose file's list.

    A directory Docker would create as root-owned at first start is harmless; one the script
    forgets is only noticed when --create is used to prepare a dataset for a different owner.
    """
    placeholder = "${APP_DATA_DIR:?set dataset path}"
    mounted = set()
    for name, service in load_compose(REPO_ROOT / "docker-compose.truenas-app.yml")["services"].items():
        for volume in service.get("volumes", []):
            source = volume.replace(placeholder, "DATA").split(":")[0]
            if source.startswith("DATA/") and name != "diun":  # the notify-only watcher's state is optional
                mounted.add(source.removeprefix("DATA/"))
    result, out = preflight(host, "--create", "--with", "canary-asr,parakeet-asr,chatterbox-tts,training,whisper-cpp")
    assert result.returncode == 0, out
    created = {str(p.relative_to(host.data_dir)) for p in host.data_dir.rglob("*")
               if p.is_dir() and not any(c.is_dir() for c in p.iterdir())}
    assert created == mounted


def test_the_data_dir_falls_back_to_the_environment(host):
    result = host.run("preflight.sh", env={"APP_DATA_DIR": str(host.data_dir)})
    assert result.returncode == 0, combined(result)


# --- Docker -------------------------------------------------------------------------------------------


def test_no_docker_is_a_failure(host):
    host.remove("docker")
    result, out = preflight(host)
    assert result.returncode == 1
    assert "docker not found" in out and "NOT READY" in out


def test_an_unreachable_daemon_is_a_failure(host):
    host.state["docker"]["usable"] = False
    result, out = preflight(host)
    assert result.returncode == 1
    assert "daemon is not reachable" in out


@pytest.mark.parametrize("version, fails", [("2.19.1", True), ("2.20.0", False), ("2.32.4", False), ("2.9.0", True)])
def test_compose_must_be_new_enough_for_depends_on_required_and_include(host, version, fails):
    host.state["docker"]["compose_version"] = version
    result, out = preflight(host)
    assert (result.returncode == 1) is fails, out
    assert fails == bool(lines_with(out, "older than 2.20.0"))


def test_a_missing_compose_plugin_is_a_failure(host):
    host.state["docker"]["compose_version"] = None
    result, out = preflight(host)
    assert result.returncode == 1 and "compose' plugin is missing" in out


# --- the NVIDIA driver -----------------------------------------------------------------------------------


def test_no_nvidia_smi_is_a_failure_that_points_at_the_driver_install(host):
    host.remove("nvidia-smi")
    result, out = preflight(host)
    assert result.returncode == 1
    assert "nvidia-smi not found" in out and "Install NVIDIA Drivers" in out


def test_no_gpu_skips_every_gpu_check(host):
    host.remove("nvidia-smi")
    result, out = preflight(host, "--no-gpu")
    assert result.returncode == 0, out
    assert "NVIDIA GPU" not in out and "VRAM budget" not in out
    assert "Planning for: frontend-service piper-tts-service" in out


def test_no_gpu_can_plan_whisper_cpp(host):
    host.remove("nvidia-smi")
    result, out = preflight(host, "--no-gpu", "--with", "whisper-cpp")
    assert result.returncode == 0, out
    assert "frontend-service piper-tts-service whisper-cpp" in out


def test_a_driver_that_reports_no_gpu_is_a_failure(host):
    host.state["nvidia"]["gpus"] = []
    result, out = preflight(host)
    assert result.returncode == 1 and "reports no GPU" in out


@pytest.mark.parametrize("driver", ["550.54.14", "545.29.06", "535.183.01", "560.35.03"])
def test_a_blackwell_card_needs_r570_whatever_the_older_branch_would_allow(host, driver):
    """535/550/560 start the CUDA 12.8 image, but not on sm_120."""
    gpu(host, driver=driver)  # the default GPU is compute capability 12.0
    host.state["nvidia"]["cuda"] = "12.4"
    result, out = preflight(host)
    assert result.returncode == 1, out
    assert "too old for a Blackwell GPU" in out and "R570" in out


@pytest.mark.parametrize("driver", ["570.86.15", "575.51.03", "580.65.06"])
def test_a_blackwell_card_on_r570_or_newer_is_fine(host, driver):
    gpu(host, driver=driver)
    result, out = preflight(host)
    assert result.returncode == 0, out
    assert "supports CUDA 12.8" in out


def test_the_blackwell_check_works_from_the_name_when_compute_cap_is_unavailable(host):
    """Older nvidia-smi cannot query compute_cap; the model name then decides."""
    host.state["nvidia"]["compute_cap"] = False
    gpu(host, driver="550.54.14", name="NVIDIA GeForce RTX 5070")
    result, out = preflight(host)
    assert result.returncode == 1 and "too old for a Blackwell GPU" in out
    gpu(host, driver="570.86.15")
    assert preflight(host)[0].returncode == 0


@pytest.mark.parametrize("branch", ["470.256.02", "535.183.01", "550.54.14", "560.35.03", "565.57.01"])
def test_a_datacentre_gpu_on_a_branch_in_the_image_allow_list_is_a_warning(host, branch):
    """nvidia/cuda:12.8.1's NVIDIA_REQUIRE_CUDA lists these branches for tesla/quadro/nvidia brands
    (CUDA forward compatibility), so the container starts; a driver with CUDA 12.8 is still safer."""
    gpu(host, name="NVIDIA A10", cc="8.6", driver=branch, mem_mib=24576)
    host.state["nvidia"]["cuda"] = "12.4"
    result, out = preflight(host)
    assert result.returncode == 0, out
    assert lines_with(out, "[WARN]") and "CUDA forward compatibility" in out


@pytest.mark.parametrize("branch", ["535.183.01", "550.54.14", "565.57.01"])
def test_a_geforce_card_needs_cuda_12_8_whatever_its_branch(host, branch):
    """The allow-list has no geforce brand: only `cuda>=12.8` can satisfy the image for a GeForce card."""
    gpu(host, name="NVIDIA GeForce RTX 3060", cc="8.6", driver=branch, mem_mib=12288)
    host.state["nvidia"]["cuda"] = "12.4"
    result, out = preflight(host)
    assert result.returncode == 1, out
    assert "accepts a GeForce card only on a driver that reports CUDA >= 12.8" in out


def test_a_datacentre_gpu_on_a_branch_outside_the_allow_list_is_a_failure(host):
    gpu(host, name="NVIDIA A10", cc="8.6", driver="545.29.06", mem_mib=24576)
    host.state["nvidia"]["cuda"] = "12.3"
    result, out = preflight(host)
    assert result.returncode == 1
    assert "refused by the CUDA 12.8 base image" in out


def test_a_driver_reporting_cuda_12_8_passes_whatever_its_branch(host):
    gpu(host, name="NVIDIA GeForce RTX 3060", cc="8.6", driver="572.16", mem_mib=12288)
    host.state["nvidia"]["cuda"] = "12.8"
    result, out = preflight(host)
    assert result.returncode == 0 and "supports CUDA 12.8" in out


def test_the_gpu_is_chosen_by_index_or_uuid_and_an_unknown_one_fails(host):
    host.state["nvidia"]["gpus"].append(
        {"index": 1, "uuid": "GPU-bbbb", "name": "NVIDIA GeForce RTX 3060", "driver": "570.86.15",
         "mem_mib": 12288, "cc": "8.6"})
    result, out = preflight(host, "--gpu-id", "1")
    assert result.returncode == 0 and "GPU 1 will be used: 12.0 GB" in out
    result, out = preflight(host, "--gpu-id", "GPU-bbbb")
    assert result.returncode == 0 and "GPU 1 will be used" in out
    result, out = preflight(host, "--gpu-id", "7")
    assert result.returncode == 1 and "no GPU with index or UUID '7'" in out


def test_a_missing_nvidia_runtime_is_only_a_warning(host):
    host.state["docker"]["runtimes"] = ["runc"]
    result, out = preflight(host)
    assert result.returncode == 0, out
    assert "does not list an nvidia runtime" in out


# --- proving the GPU inside a container ----------------------------------------------------------------------


def test_with_docker_runs_one_container_that_must_see_the_gpu(host):
    result, out = preflight(host, "--with-docker", "--gpu-id", "0")
    assert result.returncode == 0, out
    assert "a container sees the GPU: GPU 0" in out
    runs = [c for c in host.calls() if c[:2] == ["docker", "run"]]
    assert len(runs) == 1
    assert "--rm" in runs[0] and "device=0" in runs[0] and "nvidia/cuda:12.8.1-base-ubuntu22.04" in runs[0]


def test_a_container_that_prints_no_gpu_is_a_failure(host):
    host.state["docker"]["gpu_run"] = {"rc": 0, "out": "no devices"}
    result, out = preflight(host, "--with-docker")
    assert result.returncode == 1 and "printed no GPU" in out


def test_a_container_that_cannot_start_is_a_failure(host):
    host.state["docker"]["gpu_run"] = {"rc": 125, "out": 'could not select device driver "nvidia"'}
    result, out = preflight(host, "--with-docker")
    assert result.returncode == 1 and "could not run a GPU container" in out


def test_without_with_docker_no_container_is_started(host):
    preflight(host)
    assert not [c for c in host.calls() if c[:2] == ["docker", "run"]]


# --- the data directory -------------------------------------------------------------------------------------------


def test_no_data_dir_is_a_failure(host):
    result = host.run("preflight.sh")
    assert result.returncode == 1 and "no data directory given" in combined(result)


def test_a_data_dir_that_does_not_exist_is_a_failure_and_is_not_created(host):
    missing = host.root / "not-a-dataset"
    result = host.run("preflight.sh", "--data-dir", str(missing), "--create")
    out = combined(result)
    assert result.returncode == 1 and "does not exist" in out
    assert not missing.exists(), "--create makes subdirectories, never the dataset itself"


def test_the_placeholder_path_from_the_compose_file_is_flagged(host):
    result = host.run("preflight.sh", "--data-dir", "/mnt/pool/apps/tts-stt")
    assert "is the placeholder" in combined(result)


def test_a_file_system_that_is_not_zfs_warns_about_missing_snapshots(host):
    host.state["fstype"] = "ext2/ext3"
    result, out = preflight(host)
    assert result.returncode == 0, out
    assert "not ZFS" in out and "no dataset snapshots" in out


def test_on_zfs_it_prints_the_snapshot_command_for_this_dataset(host):
    _, out = preflight(host)
    assert "zfs snapshot tank/apps/tts-stt@pre-update-" in out


# --- disk space ----------------------------------------------------------------------------------------------------------


def test_less_free_space_than_the_models_need_is_a_failure(host):
    host.state["df_gb"] = {"default": 900.0, str(host.data_dir): 5.0}
    result, out = preflight(host)
    assert result.returncode == 1 and "the selected models alone need about 8.6 GB" in out


def test_room_for_the_models_but_not_for_an_update_is_a_warning(host):
    host.state["df_gb"] = {"default": 900.0, str(host.data_dir): 12.0}
    result, out = preflight(host)
    assert result.returncode == 0, out
    assert "is comfortable" in out


def test_too_little_space_for_the_images_is_a_failure(host):
    host.state["df_gb"] = {"default": 900.0, host.state["docker"]["root_dir"]: 10.0}
    result, out = preflight(host)
    assert result.returncode == 1 and "the images need about" in out


def test_when_images_and_models_share_a_file_system_the_needs_add_up(host):
    """Same device for the dataset and the Docker root: 25 GB of images plus 8.6 GB of models."""
    host.state["df_gb"] = {"default": 30.0}
    result, out = preflight(host)
    assert result.returncode == 1, out
    assert "the images need about 33.6 GB" in out


def test_optional_services_raise_the_disk_estimate(host):
    _, base = preflight(host)
    _, more = preflight(host, "--with", "chatterbox-tts,parakeet-asr")
    assert "models need about 8.6 GB" in base
    assert "models need about 14.1 GB" in more  # + 3 GB chatterbox + 2.5 GB parakeet


# --- the port ---------------------------------------------------------------------------------------------------------------


def listening_socket():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    return sock, sock.getsockname()[1]


def test_a_port_that_is_taken_is_a_failure(host):
    """Real socket, and no ss/netstat on the PATH: the bash /dev/tcp fallback answers."""
    sock, port = listening_socket()
    try:
        result, out = preflight(host, "--port", str(port))
    finally:
        sock.close()
    assert result.returncode == 1, out
    assert f"port {port} is already in use" in out


def test_a_free_port_is_fine(host):
    sock, port = listening_socket()
    sock.close()
    result, out = preflight(host, "--port", str(port))
    assert result.returncode == 0 and f"port {port} is free" in out


def test_a_port_taken_by_our_own_running_app_is_an_update_not_a_conflict(host):
    host.install("ss")
    host.state["listening"] = [3000]
    host.run_app(published_ports={"frontend-service": "0.0.0.0:3000->3000/tcp, :::3000->3000/tcp"})
    result, out = preflight(host)
    assert result.returncode == 0, out
    assert "this is an update, not a fresh install" in out


def test_a_port_taken_by_something_else_names_the_listener_when_ss_exists(host):
    host.install("ss")
    host.state["listening"] = [3000]
    result, out = preflight(host)
    assert result.returncode == 1
    assert "port 3000 is already in use" in out and "fake-listener" in out


# --- the VRAM budget -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mem_mib, verdict",
    [
        (16311, "fits with headroom"),  # 8.1 of 15.9 GB
        (12288, "fits with headroom"),  # 8.1 of 12.0 GB: under 90 percent (10.8)
        (8192, "more than 90% of 8.0 GB"),  # only works through the idle unload
    ],
)
def test_the_default_set_against_cards_of_different_size(host, mem_mib, verdict):
    gpu(host, mem_mib=mem_mib)
    result, out = preflight(host)
    assert result.returncode == 0, out
    assert verdict in out


def test_a_card_smaller_than_the_largest_single_service_fails(host):
    gpu(host, mem_mib=3072)  # qwen3-asr wants 4 GB
    result, out = preflight(host)
    assert result.returncode == 1
    assert "the largest service (qwen3-asr-service, 4 GB) does not fit" in out


def test_optional_services_are_added_to_the_total(host):
    gpu(host, mem_mib=12288)
    result, out = preflight(host, "--with", "chatterbox-tts,training")
    assert "16.1 GB" in out  # 8.1 + 4 + 4
    assert result.returncode == 0 and "more than 90%" in out


def test_a_bigger_model_choice_changes_the_budget(host):
    _, default = preflight(host)
    _, bigger = preflight(host, "--whisper-model", "large-v3", "--qwen3-tts-model", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    assert "all resident at once       8.1 GB" in default
    assert "all resident at once      11.6 GB" in bigger  # 3.1 + 4 + 4.5


def test_list_vram_prints_a_table_and_needs_no_host(host):
    for tool in ("docker", "nvidia-smi", "curl", "skopeo", "df", "stat", "findmnt"):
        host.remove(tool)
    result = host.run("preflight.sh", "--list-vram")
    rows = dict(line.split("\t") for line in result.stdout.splitlines())
    assert result.returncode == 0
    assert rows["stt-service"] == "1.6" and rows["qwen3-asr-service"] == "4" and rows["whisper-cpp"] == "0"


# --- arguments ----------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [["--bogus"], ["--with", "nonsense"], ["--port", "http"], ["--port", ""]],
)
def test_bad_arguments_exit_2_before_touching_the_host(host, args):
    result = host.run("preflight.sh", "--data-dir", str(host.data_dir), *args)
    assert result.returncode == 2, combined(result)
    assert host.calls() == []


def test_help_exits_0(host):
    result = host.run("preflight.sh", "--help")
    assert result.returncode == 0 and "--data-dir" in result.stdout
