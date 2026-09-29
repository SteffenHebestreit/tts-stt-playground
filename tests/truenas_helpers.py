"""A fake TrueNAS host for the helper scripts in scripts/truenas/.

The scripts talk to ``docker``, ``nvidia-smi``, ``curl``, ``skopeo``, ``df``, ``stat`` and
``findmnt``. Here each of those is a small stub on a PATH that holds nothing else but the
handful of real tools the scripts need (bash, awk, sed, ...), so a script runs the same on a
laptop, a CI runner and a machine with a real GPU, and a tool the scenario does not install is
genuinely absent (``command -v`` fails), which is what the "docker not found" branches need.

The stubs are one Python file (truenas_stubs.py) that reads a JSON state file, so a test
describes the host in data:

    host = FakeHost(tmp_path)
    host.state["nvidia"]["gpus"][0]["driver"] = "550.54.14"
    host.publish("0.3.0")
    result = host.run("update-check.sh", "--tag", "0.3.0")

Every stub call is appended to a log, which is how a test proves that a "read-only" script
never issued ``docker pull`` or ``docker compose up``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts" / "truenas"
STUBS_FILE = Path(__file__).resolve().with_name("truenas_stubs.py")

REGISTRY = "ghcr.io/steffenhebestreit"

# service -> image name suffix (tts-stt-<suffix>); the catalogue in lib.sh is checked against the
# compose file elsewhere, this is the fake registry's own list.
SERVICE_IMAGE = {
    "frontend-service": "frontend",
    "piper-tts-service": "piper-tts",
    "stt-service": "stt",
    "qwen3-asr-service": "qwen3-asr",
    "qwen3-tts-service": "qwen3-tts",
    "canary-asr-service": "canary-asr",
    "parakeet-asr-service": "parakeet-asr",
    "chatterbox-tts-service": "chatterbox-tts",
    "piper-training-service": "piper-training",
    "whisper-cpp": "whisper-cpp",
}
CORE = ["frontend-service", "piper-tts-service", "stt-service", "qwen3-asr-service", "qwen3-tts-service"]

# Real programs a script may call. Anything else on the fake PATH is a stub or absent.
REAL_TOOLS = (
    "bash env awk sed grep tr head tail cat dirname basename readlink mktemp rm mkdir sort uniq "
    "cut wc date sleep true false"
).split()


def digest_for(repo: str, tag: str) -> str:
    """A stable, fake manifest digest for ``repo:tag``."""
    return "sha256:" + hashlib.sha256(f"{repo}:{tag}".encode()).hexdigest()


def image_repo(service: str, registry: str = REGISTRY) -> str:
    return f"{registry}/tts-stt-{SERVICE_IMAGE[service]}"


def default_state(root: Path) -> dict:
    return {
        "docker": {
            "usable": True,
            "server_version": "27.5.1",
            "compose_version": "2.32.4",
            "runtimes": ["io.containerd.runc.v2", "nvidia", "runc"],
            "root_dir": str(root / "docker-root"),
            "containers": [],
            "gpu_run": {"rc": 0, "out": "GPU 0: NVIDIA GeForce RTX 5060 Ti (UUID: GPU-aaaa)"},
            "pull_rc": {},
        },
        "nvidia": {
            "compute_cap": True,
            "cuda": "12.8",
            "gpus": [
                {
                    "index": 0,
                    "uuid": "GPU-aaaa",
                    "name": "NVIDIA GeForce RTX 5060 Ti",
                    "driver": "570.86.15",
                    "mem_mib": 16311,
                    "cc": "12.0",
                }
            ],
        },
        # repo -> {tag: digest}
        "registry": {},
        "reachable": True,
        "methods": {"buildx": True, "skopeo": True, "curl": True},
        "df_gb": {"default": 900.0},
        "fstype": "zfs",
        "findmnt": "tank/apps/tts-stt",
        "listening": [],
        "real": {},
    }


class FakeHost:
    """A PATH of stubs plus the JSON state they answer from."""

    def __init__(self, root: Path, tools: tuple[str, ...] = ("docker", "nvidia-smi", "curl", "skopeo", "df", "stat", "findmnt")):
        self.root = Path(root)
        self.bin = self.root / "fakebin"
        self.state_file = self.root / "state.json"
        self.log_file = self.root / "calls.log"
        self.data_dir = self.root / "dataset"
        self.state = default_state(self.root)
        self.bin.mkdir(parents=True, exist_ok=True)
        (self.root / "docker-root").mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        for name in REAL_TOOLS:
            real = shutil.which(name)
            if real and not (self.bin / name).exists():
                (self.bin / name).symlink_to(real)
        self.state["real"] = {name: shutil.which(name) or "" for name in ("stat", "df", "findmnt")}
        for tool in tools:
            self.install(tool)

    # -- host description -------------------------------------------------------------------

    def install(self, tool: str) -> None:
        """Put a stub for ``tool`` on the PATH."""
        wrapper = self.bin / tool
        if wrapper.exists() or wrapper.is_symlink():
            wrapper.unlink()
        wrapper.write_text(
            f'#!/usr/bin/env bash\nexec "{sys.executable}" -S -E "{STUBS_FILE}" {tool} "$@"\n', encoding="utf-8"
        )
        wrapper.chmod(0o755)

    def remove(self, tool: str) -> None:
        """Make ``tool`` absent, as on a host that does not have it."""
        wrapper = self.bin / tool
        if wrapper.exists() or wrapper.is_symlink():
            wrapper.unlink()

    def publish(self, tag: str, services: list[str] | None = None, registry: str = REGISTRY) -> None:
        """The registry holds ``tag`` for these services (default: every image)."""
        for service in services or list(SERVICE_IMAGE):
            repo = image_repo(service, registry)
            self.state["registry"].setdefault(repo, {})[tag] = digest_for(repo, tag)

    def run_app(
        self,
        tag: str = "0.2.0",
        services: list[str] | None = None,
        *,
        project: str = "ix-tts-stt",
        stale: list[str] | None = None,
        registry: str = REGISTRY,
        published_ports: dict[str, str] | None = None,
    ) -> None:
        """Containers of an installed app. ``stale`` services run an older digest than the registry's."""
        stale = stale or []
        for service in services or CORE:
            repo = image_repo(service, registry)
            local = digest_for(repo, tag + ("-old" if service in stale else ""))
            self.state["docker"]["containers"].append(
                {
                    "name": f"{project}-{service}-1",
                    "project": project,
                    "service": service,
                    "image": f"{repo}:{tag}",
                    "image_id": "sha256:" + hashlib.sha256(f"{service}{tag}".encode()).hexdigest(),
                    "repo_digests": [f"{repo}@{local}"] if local else [],
                    "ports": (published_ports or {}).get(service, ""),
                }
            )

    def write(self) -> None:
        self.state_file.write_text(json.dumps(self.state), encoding="utf-8")

    # -- running scripts ----------------------------------------------------------------------

    def env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        env = {
            "PATH": str(self.bin),
            "HOME": str(self.root),
            "NO_COLOR": "1",
            "LC_ALL": "C",
            "FAKE_STATE": str(self.state_file),
            "FAKE_LOG": str(self.log_file),
            "TMPDIR": str(self.root),
        }
        env.update(extra or {})
        return env

    def run(
        self, script: str, *args: str, env: dict[str, str] | None = None, timeout: int = 60
    ) -> subprocess.CompletedProcess:
        """Run scripts/truenas/<script> under the fake PATH and return the finished process."""
        self.write()
        bash = shutil.which("bash")
        return subprocess.run(
            [bash, str(SCRIPTS / script), *args],
            env=self.env(env),
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(self.root),
        )

    def bash(self, snippet: str, *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
        """Run a bash snippet with lib.sh sourced (for the shared functions)."""
        self.write()
        return subprocess.run(
            [shutil.which("bash"), "-c", f'. "{SCRIPTS / "lib.sh"}"\n{snippet}'],
            env=self.env(env),
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(self.root),
        )

    def calls(self) -> list[list[str]]:
        """Every stub invocation so far, as an argv list starting with the tool name."""
        if not self.log_file.exists():
            return []
        return [json.loads(line) for line in self.log_file.read_text(encoding="utf-8").splitlines() if line]

    def docker_subcommands(self) -> set[str]:
        """The docker verbs used (``info``, ``ps``, ``image inspect``, ``buildx imagetools inspect``...)."""
        verbs = set()
        for argv in self.calls():
            if argv[0] != "docker":
                continue
            rest = [a for a in argv[1:] if not a.startswith("-") and not a.startswith("{{")]
            if rest[:1] == ["compose"]:
                verbs.add(" ".join(["compose"] + rest[1:2]))
            elif rest[:1] == ["buildx"]:
                verbs.add(" ".join(rest[:3]))
            elif rest[:1] == ["image"]:
                verbs.add(" ".join(rest[:2]))
            else:
                verbs.add(rest[0] if rest else "")
        return verbs


def combined(result: subprocess.CompletedProcess) -> str:
    return (result.stdout or "") + (result.stderr or "")
