"""The stub programs behind tests/truenas_helpers.FakeHost.

Run as ``python -S -E truenas_stubs.py <tool> args``: it answers as docker, nvidia-smi, curl,
skopeo, df, stat, findmnt or ss from the JSON state file named by FAKE_STATE and appends every
call to FAKE_LOG. Standard library only and no heavy imports, because a script under test starts
this program dozens of times.
"""

from __future__ import annotations

import json
import os
import sys

def _state() -> dict:
    with open(os.environ["FAKE_STATE"], encoding="utf-8") as fh:
        return json.load(fh)


def _log(tool: str, argv: list[str]) -> None:
    with open(os.environ["FAKE_LOG"], "a", encoding="utf-8") as fh:
        fh.write(json.dumps([tool, *argv]) + "\n")


def _flag_value(argv: list[str], flag: str) -> str | None:
    for i, item in enumerate(argv):
        if item == flag and i + 1 < len(argv):
            return argv[i + 1]
        if item.startswith(flag + "="):
            return item.split("=", 1)[1]
    return None


def _split_ref(ref: str) -> tuple[str, str]:
    ref = ref.split("@", 1)[0]
    tail = ref.rsplit("/", 1)[-1]
    if ":" in tail:
        return ref.rsplit(":", 1)[0], ref.rsplit(":", 1)[1]
    return ref, "latest"


def _registry_digest(state: dict, ref: str) -> str | None:
    if not state.get("reachable", True):
        return None
    repo, tag = _split_ref(ref)
    return (state["registry"].get(repo) or {}).get(tag)


def _registry_tags(state: dict, repo: str) -> list[str] | None:
    if not state.get("reachable", True) or repo not in state["registry"]:
        return None
    return list(state["registry"][repo])


def _docker(state: dict, argv: list[str]) -> int:
    d = state["docker"]
    fmt = _flag_value(argv, "--format") or ""
    verb = argv[0] if argv else ""
    if verb == "info":
        if not d["usable"]:
            print("Cannot connect to the Docker daemon at unix:///var/run/docker.sock", file=sys.stderr)
            return 1
        if ".DockerRootDir" in fmt:
            print(d["root_dir"])
        elif ".Runtimes" in fmt:
            print("".join(f"{r} " for r in d["runtimes"]))
        return 0
    if not d["usable"] and verb not in ("compose", "buildx"):
        print("Cannot connect to the Docker daemon", file=sys.stderr)
        return 1
    if verb == "version":
        print(d["server_version"])
        return 0
    if verb == "compose":
        if argv[1:2] == ["version"]:
            if not d.get("compose_version"):
                print("docker: 'compose' is not a docker command.", file=sys.stderr)
                return 1
            print(d["compose_version"])
            return 0
        return 1  # up / down / config are never expected from a helper script
    if verb == "ps":
        project = ""
        for i, item in enumerate(argv):
            if item == "--filter" and "com.docker.compose.project=" in argv[i + 1]:
                project = argv[i + 1].split("=", 2)[2]
        for c in d["containers"]:
            if c["project"] != project:
                continue
            print(c["ports"] if ".Ports" in fmt else c["name"])
        return 0
    if verb == "inspect":
        name = argv[-1]
        for c in d["containers"]:
            if c["name"] == name:
                print(f"{c['image']}|{c['image_id']}|{c['service']}")
                return 0
        print(f"Error: No such object: {name}", file=sys.stderr)
        return 1
    if verb == "image" and argv[1:2] == ["inspect"]:
        image_id = argv[-1]
        for c in d["containers"]:
            if c["image_id"] == image_id:
                for line in c["repo_digests"]:
                    print(line)
                return 0
        return 1
    if verb == "buildx":
        if not state["methods"]["buildx"]:
            print("docker: 'buildx' is not a docker command.", file=sys.stderr)
            return 1
        ref = argv[-1]
        digest = _registry_digest(state, ref)
        if not digest:
            print(f"ERROR: {ref}: not found", file=sys.stderr)
            return 1
        print(f"Name:      {ref}\nMediaType: application/vnd.oci.image.index.v1+json\nDigest:    {digest}")
        return 0
    if verb == "run":
        gpu = d["gpu_run"]
        print(gpu["out"])
        return gpu["rc"]
    if verb == "pull":
        ref = argv[-1]
        rc = d["pull_rc"].get(ref, 0)
        print(f"pulling {ref}" if rc == 0 else f"Error response from daemon: manifest for {ref} not found: manifest unknown")
        return rc
    print(f"fake docker: unexpected command {argv}", file=sys.stderr)
    return 1


def _nvidia_smi(state: dict, argv: list[str]) -> int:
    n = state["nvidia"]
    if not n["gpus"]:
        print("No devices were found", file=sys.stderr)
        return 6
    query = _flag_value(argv, "--query-gpu")
    if query:
        fields = query.split(",")
        if "compute_cap" in fields and not n["compute_cap"]:
            print('Field "compute_cap" is not a valid field to query.', file=sys.stderr)
            return 2
        key = {"index": "index", "uuid": "uuid", "name": "name", "driver_version": "driver",
               "memory.total": "mem_mib", "compute_cap": "cc"}
        for gpu in n["gpus"]:
            print(", ".join(str(gpu[key[f]]) for f in fields))
        return 0
    if "-L" in argv:
        for gpu in n["gpus"]:
            print(f"GPU {gpu['index']}: {gpu['name']} (UUID: {gpu['uuid']})")
        return 0
    gpu = n["gpus"][0]
    print(f"| NVIDIA-SMI {gpu['driver']}       Driver Version: {gpu['driver']}       CUDA Version: {n['cuda']}     |")
    return 0


def _curl(state: dict, argv: list[str]) -> int:
    if not state["methods"]["curl"] or not state.get("reachable", True):
        print("curl: (6) Could not resolve host", file=sys.stderr)
        return 6
    url = next((a for a in reversed(argv) if a.startswith("http")), "")
    rest = url.split("://", 1)[1]
    host, _, path = rest.partition("/")
    if path.startswith("token"):
        print('{"token":"fake-token"}')
        return 0
    if path.startswith("v2/"):
        body = path[3:]
        query = ""
        if "?" in body:
            body, query = body.split("?", 1)
        del query
        if "/manifests/" in body:
            name, tag = body.split("/manifests/", 1)
            digest = (state["registry"].get(f"{host}/{name}") or {}).get(tag)
            if not digest:
                print("curl: (22) The requested URL returned error: 404", file=sys.stderr)
                return 22
            print(f"HTTP/2 200\r\ncontent-type: application/vnd.oci.image.index.v1+json\r\ndocker-content-digest: {digest}\r\n\r")
            return 0
        if body.endswith("/tags/list"):
            name = body[: -len("/tags/list")]
            tags = _registry_tags(state, f"{host}/{name}")
            if tags is None:
                print("curl: (22) The requested URL returned error: 404", file=sys.stderr)
                return 22
            print(json.dumps({"name": name, "tags": tags}))
            return 0
    print("curl: (22) The requested URL returned error: 404", file=sys.stderr)
    return 22


def _skopeo(state: dict, argv: list[str]) -> int:
    if not state["methods"]["skopeo"]:
        print("skopeo: command not usable", file=sys.stderr)
        return 1
    target = argv[-1].removeprefix("docker://")
    if argv[0] == "inspect":
        digest = _registry_digest(state, target)
        if not digest:
            print("FATA[0000] Error parsing image name: manifest unknown", file=sys.stderr)
            return 1
        print(digest)
        return 0
    if argv[0] == "list-tags":
        tags = _registry_tags(state, target)
        if tags is None:
            print("FATA[0000] Error listing repository tags", file=sys.stderr)
            return 1
        print(json.dumps({"Repository": target, "Tags": tags}, indent=4))
        return 0
    return 1


def _df(state: dict, argv: list[str]) -> int:
    path = argv[-1]
    gb = state["df_gb"]["default"]
    for prefix, value in state["df_gb"].items():
        if prefix != "default" and path.startswith(prefix):
            gb = value
    kb = int(gb * 1024 * 1024)
    print("Filesystem     1024-blocks      Used Available Capacity Mounted on")
    print(f"fake           {kb * 2} {kb} {kb} 50% /")
    return 0


def _stat(state: dict, argv: list[str]) -> int:
    if argv[:3] == ["-f", "-c", "%T"]:
        print(state["fstype"])
        return 0
    real = state["real"]["stat"]
    sys.stdout.flush()
    os.execv(real, [real, *argv])


def _findmnt(state: dict, argv: list[str]) -> int:
    if state.get("findmnt"):
        print(state["findmnt"])
        return 0
    return 1


def _ss(state: dict, argv: list[str]) -> int:
    for item in argv:
        if item.startswith("sport = :"):
            port = int(item.split(":")[-1])
            if port in state["listening"]:
                print(f'LISTEN 0 4096 0.0.0.0:{port} 0.0.0.0:* users:(("fake-listener",pid=1,fd=3))')
            return 0
    return 0


STUBS = {
    "docker": _docker,
    "nvidia-smi": _nvidia_smi,
    "curl": _curl,
    "skopeo": _skopeo,
    "df": _df,
    "stat": _stat,
    "findmnt": _findmnt,
    "ss": _ss,
}


def _stub_main(tool: str, argv: list[str]) -> int:
    _log(tool, argv)
    return STUBS[tool](_state(), argv)


if __name__ == "__main__":
    sys.exit(_stub_main(sys.argv[1], sys.argv[2:]))
