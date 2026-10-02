"""settings_store.py: the files behind the Settings page.

What has to hold for the settings to survive and for the API never to open by accident:
writes are atomic and serialized across the uvicorn workers, a stale page cannot overwrite
a newer save, a damaged preferences file falls back softly while a damaged keys file fails
closed, nothing is written before the first save, and no key or claim code is ever stored
in the clear. Numbers in the test names refer to the store tests of the phase-1 plan.
"""

from __future__ import annotations

import base64
import errno
import hashlib
import importlib
import json
import logging
import os
import re
import secrets
import shutil
import sys
import threading
from pathlib import Path

import pytest

from frontend_loader import SERVICE_DIR


def _import(name: str):
    sys.path.insert(0, str(SERVICE_DIR))
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(SERVICE_DIR))


store_module = _import("settings_store")
SettingsStore = store_module.SettingsStore
Actor = store_module.Actor

ACTOR = Actor(ip="192.168.1.20", host="truenas.k2o", credential=store_module.DEPLOYMENT_KEY)
POSIX_ONLY = pytest.mark.skipif(os.name != "posix", reason="POSIX file modes and symbolic links")


class FakeClock:
    def __init__(self, start: float = 1_790_000_000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def new_key() -> str:
    return "tts_" + base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii").rstrip("=")


def everything_in(folder: Path) -> str:
    return "\n".join(path.read_bytes().decode("utf-8", "replace") for path in sorted(folder.rglob("*"))
                     if path.is_file())


def temp_files(folder: Path) -> list[str]:
    return sorted(path.name for path in folder.rglob("*") if path.name.endswith(".tmp"))


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def folder(tmp_path):
    path = tmp_path / "settings"
    path.mkdir()
    return path


@pytest.fixture
def lock_path(tmp_path):
    return str(tmp_path / "settings.lock")


@pytest.fixture
def make_store(folder, lock_path, clock):
    def make(directory=None, **kwargs):
        kwargs.setdefault("lock_path", lock_path)
        kwargs.setdefault("clock", clock)
        kwargs.setdefault("verify_mount", False)
        return SettingsStore(folder if directory is None else directory, **kwargs)

    return make


def write_json(path: Path, document) -> None:
    path.write_text(json.dumps(document), encoding="utf-8")


# --- 1. atomic writes ----------------------------------------------------------------------

def test_1_a_write_is_atomic_and_leaves_no_temp_file(make_store, folder, monkeypatch):
    store = make_store()
    store.update_preferences({"MAX_TTS_CHARS": 4321}, base_revision=0, actor=ACTOR)
    target = folder / "gateway.json"
    seen = []
    real_replace = os.replace

    def watching_replace(source, destination, *args, **kwargs):
        if Path(destination) == target:
            seen.append((Path(source).parent == target.parent,
                         json.loads(Path(source).read_text(encoding="utf-8"))["revision"],
                         json.loads(target.read_text(encoding="utf-8"))["revision"]))
        return real_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "replace", watching_replace)
    store.update_preferences({"MAX_TTS_CHARS": 3000}, base_revision=1, actor=ACTOR)
    # The new file was complete next to the target, and the old one was still whole, when it was swapped in.
    assert seen == [(True, 2, 1)]
    assert json.loads(target.read_text(encoding="utf-8"))["revision"] == 2
    assert temp_files(folder) == []

    # A failure half-way (here: fsync) leaves the previous file and no temp file behind.
    before = target.read_bytes()

    def failing_fsync(fd):
        raise OSError(5, "I/O error")

    monkeypatch.setattr(os, "fsync", failing_fsync)
    with pytest.raises(store_module.WriteFailed) as info:
        store.update_preferences({"MAX_TTS_CHARS": 2000}, base_revision=2, actor=ACTOR)
    assert info.value.code == "settings_write_failed"
    assert target.read_bytes() == before
    assert temp_files(folder) == []
    assert store.state.preferences.values["MAX_TTS_CHARS"] == 3000


# --- 2. history -----------------------------------------------------------------------------

def test_2_history_keeps_the_last_20_versions(make_store, folder, clock):
    store = make_store()
    for revision in range(25):
        store.update_preferences({"MAX_TTS_CHARS": 1000 + revision}, base_revision=revision, actor=ACTOR)
        clock.advance(1)
    files = sorted((folder / "history").iterdir())
    assert len(files) == 20
    assert all(re.fullmatch(r"\d{8}T\d{6}Z-r\d+\.json", path.name) for path in files)
    entries = store.history()
    assert [entry.revision for entry in entries] == list(range(25, 5, -1))
    newest = entries[0]
    assert newest.event == "save" and newest.changed == ("MAX_TTS_CHARS",)
    assert newest.saved_by["credential"] == "deployment key" and newest.saved_by["host"] == "truenas.k2o"
    assert newest.as_dict()["id"] == newest.id
    snapshot = json.loads((folder / "history" / f"{newest.id}.json").read_text(encoding="utf-8"))
    assert snapshot["values"] == {"MAX_TTS_CHARS": 1024}
    assert len(store.history(limit=5)) == 5


# --- 3. revisions ---------------------------------------------------------------------------

def test_3_a_stale_revision_is_a_conflict_and_changes_nothing(make_store, folder):
    first = make_store()
    stale = make_store()
    assert stale.state.preferences.revision == 0
    first.update_preferences({"MAX_TTS_CHARS": 4000}, base_revision=0, actor=ACTOR)
    before = (folder / "gateway.json").read_bytes()
    history_before = sorted(os.listdir(folder / "history"))
    audit_before = (folder / "audit.jsonl").read_bytes()

    with pytest.raises(store_module.RevisionConflict) as info:
        stale.update_preferences({"MAX_TTS_CHARS": 3000}, base_revision=0, actor=ACTOR)
    assert info.value.current_revision == 1 and info.value.code == "revision_conflict"
    assert (folder / "gateway.json").read_bytes() == before
    assert sorted(os.listdir(folder / "history")) == history_before
    assert (folder / "audit.jsonl").read_bytes() == audit_before
    # The refused instance now knows the current state.
    assert stale.state.preferences.values["MAX_TTS_CHARS"] == 4000


# --- 4. the lock ----------------------------------------------------------------------------

def test_4_two_store_instances_are_serialized_by_the_lock(make_store):
    first, second = make_store(), make_store()
    inside, release = threading.Event(), threading.Event()
    results = {}

    def hold(_values, _changed):
        inside.set()
        assert release.wait(10)

    def run_first():
        results["first"] = first.update_preferences({"MAX_TTS_CHARS": 4000}, base_revision=0, actor=ACTOR,
                                                     check=hold)

    def run_second():
        try:
            results["second"] = second.update_preferences({"MAX_TTS_CHARS": 3000}, base_revision=0, actor=ACTOR)
        except store_module.RevisionConflict as exc:
            results["second"] = exc

    one = threading.Thread(target=run_first)
    one.start()
    assert inside.wait(10)
    two = threading.Thread(target=run_second)
    two.start()
    two.join(0.5)
    assert two.is_alive(), "the second writer did not wait for the first one's lock"
    release.set()
    one.join(10)
    two.join(10)
    assert results["first"].revision == 1
    # It re-read the file under the lock, so it saw revision 1 instead of overwriting it.
    assert isinstance(results["second"], store_module.RevisionConflict)
    assert results["second"].current_revision == 1


def test_4_the_lock_is_exclusive_between_holders_and_reentrant_within_one(lock_path):
    holder = store_module.FileLock(lock_path, timeout=5)
    other = store_module.FileLock(lock_path, timeout=0.2)
    with holder:
        with pytest.raises(store_module.LockTimeout):
            other.acquire()
        with holder:  # re-entrant
            pass
        with pytest.raises(store_module.LockTimeout):
            other.acquire()
    with other:
        pass


# --- 5. file modes --------------------------------------------------------------------------

@POSIX_ONLY
def test_5_files_are_0600_and_folders_0700(make_store, folder, lock_path):
    previous = os.umask(0o022)
    try:
        store = make_store()
        store.update_preferences({"TRUSTED_HOSTS": ["speach.k2o"]}, base_revision=0, actor=ACTOR)
        store.add_key(name="Home Assistant", role="client", key=new_key(), actor=ACTOR)
        store.issue_code(actor=ACTOR)
    finally:
        os.umask(previous)
    files = [folder / "gateway.json", folder / "keys.json", folder / "codes.json", folder / "audit.jsonl",
             *(folder / "history").iterdir(), Path(lock_path)]
    for path in files:
        assert path.stat().st_mode & 0o777 == 0o600, path.name
    assert (folder / "history").stat().st_mode & 0o777 == 0o700


# --- 6. change detection --------------------------------------------------------------------

def test_6_refresh_notices_a_replacement_and_rereads_nothing_otherwise(make_store, folder):
    writer, reader = make_store(), make_store()
    assert reader.refresh() is True  # the first look
    reads = reader.reads
    assert reader.refresh() is False and reader.reads == reads

    writer.update_preferences({"MAX_TTS_CHARS": 4321}, base_revision=0, actor=ACTOR)
    unchanged_keys = reader.state.keys
    assert reader.refresh() is True and reader.reads == reads + 1
    assert reader.state.preferences.values["MAX_TTS_CHARS"] == 4321
    assert reader.state.keys is unchanged_keys  # only what changed is replaced
    assert reader.refresh() is False and reader.reads == reads + 1

    # Same size, same mtime, other content: the inode of the os.replace() still gives it away.
    target = folder / "gateway.json"
    old = os.stat(target)
    data = target.read_bytes()
    assert data.count(b"4321") == 1
    temp = folder / "hand.tmp"
    temp.write_bytes(data.replace(b"4321", b"4322"))
    os.utime(temp, ns=(old.st_atime_ns, old.st_mtime_ns))
    os.replace(temp, target)
    new = os.stat(target)
    assert (new.st_size, new.st_mtime_ns) == (old.st_size, old.st_mtime_ns) and new.st_ino != old.st_ino
    assert reader.refresh() is True
    assert reader.state.preferences.values["MAX_TTS_CHARS"] == 4322


def test_6_safe_mode_is_noticed(make_store, folder):
    store = make_store()
    assert store.state.safe_mode is False
    (folder / "SAFE-MODE").write_text("", encoding="utf-8")
    assert store.refresh() is True and store.state.safe_mode is True
    (folder / "SAFE-MODE").unlink()
    assert store.refresh() is True and store.state.safe_mode is False


# --- 7. lenient reading ---------------------------------------------------------------------

def test_7_invalid_values_are_dropped_with_a_reason_and_unknown_keys_ignored(make_store, folder):
    write_json(folder / "gateway.json", {
        "schema": 3, "revision": 7, "saved_at": "2026-10-02T09:12:03Z",
        "saved_by": {"ip": "192.168.1.20", "host": "truenas.k2o", "credential": "deployment key", "x": 1},
        "values": {"MAX_TTS_CHARS": "lots", "TRUSTED_HOSTS": ["speach.k2o", "*.de"], "MAX_UPLOAD_MB": 100,
                   "A_SETTING_FROM_A_NEWER_RELEASE": True, "require_key": True},
        "pending": None, "something_else": [1, 2],
    })
    preferences = make_store().state.preferences
    assert dict(preferences.values) == {"MAX_UPLOAD_MB": 100}
    assert set(preferences.dropped) == {"MAX_TTS_CHARS", "TRUSTED_HOSTS"}
    assert "anyone can register" in preferences.dropped["TRUSTED_HOSTS"]
    assert any(problem.startswith("MAX_TTS_CHARS") for problem in preferences.problems)
    assert preferences.revision == 7 and preferences.source == "file" and not preferences.damaged
    assert dict(preferences.saved_by) == {"ip": "192.168.1.20", "host": "truenas.k2o", "credential": "deployment key"}

    # A save keeps only what is valid.
    store = make_store()
    store.update_preferences({"MAX_TTS_CHARS": 3000}, base_revision=7, actor=ACTOR)
    assert json.loads((folder / "gateway.json").read_text(encoding="utf-8"))["values"] == {
        "MAX_UPLOAD_MB": 100, "MAX_TTS_CHARS": 3000}


def test_7_a_broken_pending_record_is_ignored_not_fatal(make_store, folder):
    write_json(folder / "gateway.json", {"schema": 1, "revision": 2, "values": {"TRUST_PROXY_HEADERS": True},
                                         "pending": {"revision": "two"}})
    preferences = make_store().state.preferences
    assert preferences.pending is None and preferences.values["TRUST_PROXY_HEADERS"] is True
    assert any("pending" in problem for problem in preferences.problems)


# --- 8. a damaged gateway.json --------------------------------------------------------------

def test_8_a_damaged_file_falls_back_to_last_good_then_history_then_the_yaml(make_store, folder):
    running = make_store()
    running.update_preferences({"MAX_TTS_CHARS": 4000}, base_revision=0, actor=ACTOR)
    (folder / "gateway.json").write_text("{ not json", encoding="utf-8")

    # A running worker keeps what it had.
    assert running.refresh() is True
    preferences = running.state.preferences
    assert preferences.damaged and preferences.source == "last_good"
    assert dict(preferences.values) == {"MAX_TTS_CHARS": 4000} and preferences.revision == 1
    assert "damaged" in preferences.problems[0]

    # A worker that starts now takes the newest valid history version, skipping damaged ones.
    (folder / "history" / "20991231T000000Z-r99.json").write_text("garbage", encoding="utf-8")
    preferences = make_store().state.preferences
    assert preferences.source == "history" and preferences.damaged
    assert dict(preferences.values) == {"MAX_TTS_CHARS": 4000} and preferences.revision == 1

    # Without any history: the app YAML values.
    shutil.rmtree(folder / "history")
    fresh = make_store()
    preferences = fresh.state.preferences
    assert preferences.source == "none" and preferences.damaged and dict(preferences.values) == {}

    # Saving repairs the file and keeps the damaged one for inspection.
    fresh.update_preferences({"MAX_TTS_CHARS": 3000}, base_revision=0, actor=ACTOR)
    assert (folder / "gateway.json.damaged").read_text(encoding="utf-8") == "{ not json"
    assert not fresh.state.preferences.damaged and fresh.state.preferences.source == "file"
    assert running.refresh() is True and not running.state.preferences.damaged


def test_8_a_file_that_was_absent_stays_absent_in_force_when_a_broken_one_appears(make_store, folder):
    store = make_store()
    assert store.state.preferences.source == "absent"
    (folder / "gateway.json").write_text("[]", encoding="utf-8")
    assert store.refresh() is True
    preferences = store.state.preferences
    assert preferences.source == "last_good" and preferences.damaged and dict(preferences.values) == {}
    assert "app YAML" in preferences.problems[0]


@POSIX_ONLY
def test_8_a_symlinked_or_special_gateway_json_is_not_followed(make_store, folder, tmp_path):
    elsewhere = tmp_path / "elsewhere.json"
    write_json(elsewhere, {"schema": 1, "revision": 1, "values": {"MAX_TTS_CHARS": 10}})
    (folder / "gateway.json").symlink_to(elsewhere)
    preferences = make_store().state.preferences
    assert preferences.damaged and dict(preferences.values) == {}
    (folder / "gateway.json").unlink()
    fifo = folder / "gateway.json"
    os.mkfifo(fifo)
    seen = {}
    reader = threading.Thread(target=lambda: seen.update(preferences=make_store().state.preferences), daemon=True)
    reader.start()
    reader.join(10)
    if reader.is_alive():
        # Release the reader blocked in open() so the suite does not hang, then fail.
        os.close(os.open(fifo, os.O_WRONLY))
        reader.join(10)
        pytest.fail("reading settings blocked on a FIFO")
    preferences = seen["preferences"]
    assert preferences.damaged and "regular" in preferences.problems[0]


# --- 9. a damaged keys.json ------------------------------------------------------------------

def _valid_keys_document(**overrides):
    document = {"schema": 1, "revision": 3, "require_key": False, "deployment_key_role": "admin", "keys": [
        {"id": "9f2c1a7b", "name": "Home Assistant", "role": "client", "sha256": "a" * 64, "hint": "tts_Ab3d",
         "created_at": "2026-10-02T09:12:03Z"}]}
    document.update(overrides)
    return document


@pytest.mark.parametrize("content", [
    "not json",
    '{"schema": 1, "schema": 1, "revision": 1, "require_key": false, "deployment_key_role": "admin", "keys": []}',
    json.dumps(_valid_keys_document(schema=2)),
    json.dumps(_valid_keys_document(require_key="yes")),
    json.dumps(_valid_keys_document(deployment_key_role="root")),
    json.dumps(_valid_keys_document(keys=[{"id": "1", "name": "x", "role": "admin", "sha256": "nothex"}])),
    json.dumps(_valid_keys_document(keys=[{"id": "1", "name": "x", "role": "admin", "sha256": "b" * 64},
                                          {"id": "1", "name": "y", "role": "client", "sha256": "c" * 64}])),
    json.dumps({"keys": []}),
    "",
], ids=["not-json", "duplicate-key", "newer-schema", "bad-require", "bad-role", "bad-digest", "duplicate-id",
        "incomplete", "empty"])
def test_9_a_damaged_keys_file_fails_closed(make_store, folder, content):
    (folder / "keys.json").write_text(content, encoding="utf-8")
    keys = make_store().state.keys
    assert keys.fail_closed and keys.require_key and keys.keys == ()
    assert keys.deployment_key_role == "client"
    assert keys.problem and "damaged" in keys.problem
    assert keys.effective_require_key(False) is True
    assert keys.has_admin(True) is False  # not even the YAML key may change settings: recovery only


def test_9_a_running_worker_fails_closed_too_and_only_a_claim_repairs(make_store, folder):
    store = make_store()
    admin_key = new_key()
    admin = store.add_key(name="Admin", role="admin", key=admin_key, actor=ACTOR)
    assert store.state.keys.find(admin_key) == admin
    (folder / "keys.json").write_text("{", encoding="utf-8")
    assert store.refresh() is True
    keys = store.state.keys
    assert keys.fail_closed and keys.find(admin_key) is None
    with pytest.raises(store_module.KeysFileDamaged):
        store.add_key(name="Other", role="admin", key=new_key(), actor=ACTOR)
    with pytest.raises(store_module.KeysFileDamaged):
        store.revoke_key(admin.id, actor=ACTOR, deployment_key_present=True)
    with pytest.raises(store_module.KeysFileDamaged) as info:
        store.set_access(base_revision=keys.revision, actor=ACTOR, deployment_key_present=True, require_key=True)
    assert info.value.code == "keys_file_damaged"
    assert (folder / "keys.json").read_text(encoding="utf-8") == "{"


@POSIX_ONLY
def test_9_a_symlinked_keys_file_fails_closed(make_store, folder, tmp_path):
    elsewhere = tmp_path / "keys.json"
    write_json(elsewhere, _valid_keys_document())
    (folder / "keys.json").symlink_to(elsewhere)
    assert make_store().state.keys.fail_closed


def test_9_a_valid_keys_file_is_read(make_store, folder):
    write_json(folder / "keys.json", _valid_keys_document())
    keys = make_store().state.keys
    assert not keys.fail_closed and keys.revision == 3 and keys.require_key is False
    assert [record.public() for record in keys.keys] == [
        {"id": "9f2c1a7b", "name": "Home Assistant", "role": "client", "hint": "tts_Ab3d",
         "created_at": "2026-10-02T09:12:03Z"}]
    assert "sha256" not in json.dumps(keys.public())


# --- 10. nothing is written before the first save ------------------------------------------------

def test_10_constructing_and_reading_write_nothing(folder, tmp_path, monkeypatch):
    lock = tmp_path / "lock"
    store = SettingsStore(folder, verify_mount=False, lock_path=str(lock))
    assert list(folder.iterdir()) == [] and not lock.exists()
    store.refresh()
    _ = store.state
    store.mount_status()
    store.history()
    assert store.expire_pending() is False
    assert list(folder.iterdir()) == [] and not lock.exists()

    monkeypatch.setenv("TTS_STT_SETTINGS_DIR", str(folder))
    from_env = SettingsStore.from_environment(lock_path=str(lock))
    assert from_env.directory == folder and from_env.verify_mount is False
    from_env.refresh()
    assert list(folder.iterdir()) == [] and not lock.exists()

    monkeypatch.delenv("TTS_STT_SETTINGS_DIR")
    default = SettingsStore.from_environment(lock_path=str(lock))
    assert default.directory == Path("/app/settings") and default.verify_mount is True
    assert SettingsStore.from_environment({"TTS_STT_SETTINGS_DIR": " "}).verify_mount is True
    assert SettingsStore.from_environment({"TTS_STT_SETTINGS_DIR": str(folder)}).directory == folder


def test_10_the_store_never_creates_its_folder(tmp_path, lock_path):
    missing = tmp_path / "not-there"
    store = SettingsStore(missing, verify_mount=False, lock_path=lock_path)
    status = store.mount_status()
    assert status.state == "missing" and not status.writable
    with pytest.raises(store_module.NotWritable):
        store.update_preferences({"MAX_TTS_CHARS": 4000}, base_revision=0, actor=ACTOR)
    with pytest.raises(store_module.NotWritable):
        store.issue_code(actor=ACTOR)
    store.audit("refused", actor=ACTOR)
    assert not missing.exists() and not Path(lock_path).exists()


# --- 11. the mount table ---------------------------------------------------------------------

MOUNTINFO = (
    "22 1 0:21 / / rw,relatime master:1 - overlay overlay rw,lowerdir=/var/lib/docker/overlay2/l/A:/var/lib/"
    "docker/overlay2/l/B,upperdir=/var/lib/docker/overlay2/x/diff,workdir=/var/lib/docker/overlay2/x/work\n"
    "23 22 0:22 / /proc rw,nosuid,nodev,noexec,relatime - proc proc rw\n"
    "24 22 0:23 / /dev rw,nosuid - tmpfs tmpfs rw,size=65536k,mode=755\n"
    "31 22 0:45 /settings /app/settings rw,relatime shared:9 master:3 - zfs MainPool/Apps/tts-stt rw,xattr,posixacl\n"
    "32 22 0:45 /backend-settings /app/backend-settings ro,relatime - zfs MainPool/Apps/tts-stt rw,xattr\n"
    "33 22 8:1 /var/lib/docker/containers/c0ffee/hosts /etc/hosts rw,relatime - ext4 /dev/sda1 rw\n"
    "34 22 0:50 /mnt/a\\040b /app/with\\040space rw - 9p drvfs rw,dirsync,aname=drvfs;path=C:\\134\n"
    "this line is garbage\n"
    "\n"
)


def _escape(path: str) -> str:
    return "".join(f"\\{ord(ch):03o}" if ch in " \t\n\\" else ch for ch in path)


def test_11_the_mountinfo_parser():
    entries = store_module.parse_mountinfo(MOUNTINFO)
    assert len(entries) == 7
    settings = store_module.find_mount(entries, "/app/settings")
    assert (settings.mount_id, settings.parent_id, settings.root) == (31, 22, "/settings")
    assert (settings.fs_type, settings.source) == ("zfs", "MainPool/Apps/tts-stt")
    assert not settings.read_only and "relatime" in settings.options
    assert store_module.find_mount(entries, "/app/backend-settings").read_only
    assert store_module.find_mount(entries, "/app/settings/") == settings
    assert store_module.find_mount(entries, "/app") is None
    assert store_module.find_mount(entries, "/app/settings/history") is None
    spaced = store_module.find_mount(entries, "/app/with space")
    assert spaced.root == "/mnt/a b" and spaced.fs_type == "9p"
    assert store_module.parse_mountinfo("garbage\n1 2 3\nx y 0:1 / /a rw - ext4 /dev/x rw\n") == []
    # Mounted twice at the same point: the later (topmost) one counts.
    stacked = MOUNTINFO + "40 22 0:60 / /app/settings ro - tmpfs tmpfs ro\n"
    assert store_module.find_mount(store_module.parse_mountinfo(stacked), "/app/settings").read_only


def test_11_only_a_read_write_mount_point_is_writable(folder, tmp_path, lock_path, clock):
    def store_with(text: str) -> SettingsStore:
        info = tmp_path / "mountinfo"
        info.write_text(text, encoding="utf-8")
        return SettingsStore(folder, verify_mount=True, mountinfo_path=str(info), lock_path=lock_path, clock=clock)

    line = f"40 22 0:45 /settings {_escape(os.fspath(folder))} rw,relatime - zfs pool rw\n"
    mounted = store_with(MOUNTINFO + line)
    assert mounted.mount_status().state == "mounted" and mounted.mount_status().writable

    for text, state in ((MOUNTINFO + line.replace(" rw,relatime ", " ro,relatime "), "read_only"),
                        (MOUNTINFO, "not_mounted"), ("", "not_mounted")):
        store = store_with(text)
        status = store.mount_status()
        assert status.state == state and not status.writable and status.detail
        with pytest.raises(store_module.NotWritable) as info:
            store.update_preferences({"MAX_TTS_CHARS": 4000}, base_revision=0, actor=ACTOR)
        assert info.value.code == "settings_not_mounted"

    unreadable = SettingsStore(folder, verify_mount=True, mountinfo_path=str(tmp_path / "missing"), lock_path=lock_path)
    assert unreadable.mount_status().state == "not_mounted"
    assert list(folder.iterdir()) == []

    # Without the mount, what is there is still read (and shown read-only).
    write_json(folder / "gateway.json", {"schema": 1, "revision": 4, "values": {"MAX_TTS_CHARS": 900}})
    assert store_with(MOUNTINFO).state.preferences.values["MAX_TTS_CHARS"] == 900

    mounted.update_preferences({"MAX_TTS_CHARS": 4000}, base_revision=4, actor=ACTOR)
    assert json.loads((folder / "gateway.json").read_text(encoding="utf-8"))["revision"] == 5


# --- 12. one-time codes ------------------------------------------------------------------------

CODE_SHAPE = re.compile(r"[0-9A-HJKMNP-TV-Z]{5}(-[0-9A-HJKMNP-TV-Z]{5}){3}")


def test_12_codes_are_hashed_single_use_and_expire(make_store, folder, clock):
    store = make_store()
    code = store.issue_code(actor=ACTOR)
    assert CODE_SHAPE.fullmatch(code)
    canonical = code.replace("-", "")
    stored = (folder / "codes.json").read_text(encoding="utf-8")
    assert code not in stored and canonical not in stored
    assert hashlib.sha256(canonical.encode("ascii")).hexdigest() in stored
    assert code not in everything_in(folder)

    # Misreadings are forgiven: lower case, spaces for dashes, O for 0, I and L for 1.
    variant = code.lower().replace("-", " ").replace("0", "o").replace("1", "l")
    assert store.redeem_code(variant, actor=ACTOR) is True
    assert store.redeem_code(code, actor=ACTOR) is False  # single use

    clock.advance(61)
    second = store.issue_code(actor=ACTOR)
    clock.advance(30 * 60 + 1)
    assert store.redeem_code(second, actor=ACTOR) is False  # expired


def test_12_wrong_guesses_cancel_no_code(make_store, folder, clock):
    """A wrong guess cannot be told apart by the code it was aimed at, so it costs no code anything:
    a budget shared by every caller would let anyone on the network cancel the owner's code. The
    gateway limits wrong codes per address instead."""
    store, stranger = make_store(), make_store()
    owner = Actor(ip="192.168.1.20", host="192.168.1.20:3000", credential="one-time code")
    other = Actor(ip="192.168.1.66", host="192.168.1.20:3000", credential="one-time code")
    code = store.issue_code(actor=owner)
    clock.advance(61)
    second = store.issue_code(actor=owner)
    codes_before = (folder / "codes.json").read_bytes()
    wrong = ["00000-00000-00000-00000", "AAAAA-AAAAA-AAAAA-AAAAA", "not even the right shape", "x", "", None, 123,
             "-" * 20]
    for guess in wrong * 3:
        assert stranger.redeem_code(guess, actor=other) is False
        with pytest.raises(store_module.InvalidCode):
            stranger.claim(code=guess, name="Intruder", key=new_key(), actor=other)
    assert (folder / "codes.json").read_bytes() == codes_before, "a wrong guess changed the codes"
    audit = [json.loads(line) for line in (folder / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sum(1 for line in audit if line["event"] == "code-refused" and line["ip"] == "192.168.1.66") == 48

    record = store.claim(code=code, name="Owner", key=new_key(), actor=owner)
    assert record.role == "admin"
    with pytest.raises(store_module.InvalidCode):
        store.claim(code=code, name="Again", key=new_key(), actor=owner)      # still single use
    assert store.redeem_code(second, actor=owner) is True                    # the other one is untouched
    assert "attempts_left" not in (folder / "codes.json").read_text(encoding="utf-8")


def test_12_codes_written_by_the_earlier_layout_still_work(make_store, folder, clock):
    """codes.json written before attempts were dropped carries "attempts_left"; it is ignored."""
    store = make_store()
    code = store.issue_code(actor=ACTOR)
    document = json.loads((folder / "codes.json").read_text(encoding="utf-8"))
    for record in document["codes"]:
        record["attempts_left"] = 1
    write_json(folder / "codes.json", document)
    assert store.redeem_code("00000-00000-00000-00000", actor=ACTOR) is False
    assert store.redeem_code(code, actor=ACTOR) is True


def test_12_codes_are_capped_and_rate_limited(make_store, clock):
    store = make_store()
    first = store.issue_code(actor=ACTOR)
    with pytest.raises(store_module.RateLimited) as info:
        store.issue_code(actor=ACTOR)
    assert 1 <= info.value.retry_after <= 60 and info.value.code == "rate_limited"
    codes = [first]
    for _ in range(2):
        clock.advance(61)
        codes.append(store.issue_code(actor=ACTOR))
    clock.advance(61)
    with pytest.raises(store_module.RateLimited) as info:
        store.issue_code(actor=ACTOR)
    assert info.value.retry_after > 60  # until the oldest one expires
    assert all(store.redeem_code(code, actor=ACTOR) for code in codes)  # every one still works
    clock.advance(61)
    assert CODE_SHAPE.fullmatch(store.issue_code(actor=ACTOR))


def test_12_a_damaged_codes_file_holds_no_valid_code(make_store, folder):
    store = make_store()
    code = store.issue_code(actor=ACTOR)
    (folder / "codes.json").write_text("{]", encoding="utf-8")
    assert store.redeem_code(code, actor=ACTOR) is False


# --- commit-confirm --------------------------------------------------------------------------

def test_a_proxy_key_change_is_live_at_once_and_confirm_clears_it(make_store, clock):
    store, other = make_store(), make_store()
    result = store.update_preferences({"TRUSTED_ORIGINS": ["https://tts.example.com"]}, base_revision=0, actor=ACTOR)
    assert result.written and result.revision == 1
    assert result.pending.revision == 1 and result.pending.deadline == clock.now + 60
    assert dict(result.pending.previous) == {"TRUSTED_ORIGINS": None}
    assert store.state.preferences.values["TRUSTED_ORIGINS"] == ("https://tts.example.com",)
    other.refresh()
    assert other.state.preferences.pending.revision == 1

    assert store.confirm(0, actor=ACTOR) is False  # not that revision
    assert store.confirm(1, actor=ACTOR) is True
    assert store.state.preferences.pending is None and store.state.preferences.revision == 1
    assert other.refresh() is True and other.state.preferences.pending is None
    assert store.confirm(1, actor=ACTOR) is False  # nothing left to confirm
    # Keys that need no confirmation get no record.
    assert store.update_preferences({"MAX_TTS_CHARS": 4000}, base_revision=1, actor=ACTOR).pending is None


def test_an_unconfirmed_change_is_undone_after_the_deadline_by_any_worker(make_store, folder, clock):
    writer, other = make_store(), make_store()
    writer.update_preferences({"TRUSTED_ORIGINS": ["https://old.example.com"]}, base_revision=0, actor=ACTOR)
    assert writer.confirm(1, actor=ACTOR)
    result = writer.update_preferences({"TRUSTED_ORIGINS": ["https://new.example.com"], "TRUST_PROXY_HEADERS": True,
                                        "MAX_TTS_CHARS": 4000}, base_revision=1, actor=ACTOR)
    assert dict(result.pending.previous) == {"TRUST_PROXY_HEADERS": None,
                                             "TRUSTED_ORIGINS": ("https://old.example.com",)}

    clock.advance(59)
    other.refresh()
    assert other.expire_pending() is False
    clock.advance(1)
    other.refresh()
    assert other.expire_pending() is True
    preferences = other.state.preferences
    assert preferences.values["TRUSTED_ORIGINS"] == ("https://old.example.com",)
    assert "TRUST_PROXY_HEADERS" not in preferences.values
    assert preferences.values["MAX_TTS_CHARS"] == 4000  # the rest of the save stays
    assert preferences.pending is None and preferences.revision == 3

    assert writer.refresh() is True and writer.state.preferences.revision == 3
    assert writer.expire_pending() is False
    assert writer.history()[0].event == "auto-revert"
    audit = [json.loads(line) for line in (folder / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
    assert audit[-1]["event"] == "auto-reverted" and audit[-1]["reverted_revision"] == 2


def test_a_pending_record_found_at_start_is_undone(make_store, folder, clock):
    write_json(folder / "gateway.json", {
        "schema": 1, "revision": 5, "values": {"TRUST_PROXY_HEADERS": True, "MAX_UPLOAD_MB": 100},
        "pending": {"revision": 5, "deadline": clock.now - 5, "previous": {"TRUST_PROXY_HEADERS": None}}})
    store = make_store()
    assert store.state.preferences.pending is not None
    assert store.expire_pending() is True
    assert dict(store.state.preferences.values) == {"MAX_UPLOAD_MB": 100}


def test_a_late_confirm_does_not_keep_the_change(make_store, clock):
    store = make_store()
    store.update_preferences({"TRUST_PROXY_HEADERS": True}, base_revision=0, actor=ACTOR)
    clock.advance(61)
    assert store.confirm(1, actor=ACTOR) is False
    assert "TRUST_PROXY_HEADERS" not in store.state.preferences.values


def test_later_saves_carry_the_record_and_undoing_the_change_drops_it(make_store, clock):
    store = make_store()
    first = store.update_preferences({"TRUST_PROXY_HEADERS": True}, base_revision=0, actor=ACTOR)
    clock.advance(10)
    second = store.update_preferences({"MAX_TTS_CHARS": 4000}, base_revision=1, actor=ACTOR)
    assert second.pending == first.pending  # carried over, same deadline
    clock.advance(10)
    third = store.update_preferences({}, reset=["TRUST_PROXY_HEADERS"], base_revision=2, actor=ACTOR)
    assert third.pending is None and store.state.preferences.pending is None


def test_a_save_after_the_deadline_first_undoes_then_reports_the_new_revision(make_store, clock):
    store = make_store()
    store.update_preferences({"TRUST_PROXY_HEADERS": True}, base_revision=0, actor=ACTOR)
    clock.advance(120)
    with pytest.raises(store_module.RevisionConflict) as info:
        store.update_preferences({"MAX_TTS_CHARS": 4000}, base_revision=1, actor=ACTOR)
    assert info.value.current_revision == 2
    assert "TRUST_PROXY_HEADERS" not in store.state.preferences.values


# --- writing preferences ---------------------------------------------------------------------

def test_set_reset_and_no_op(make_store, folder):
    store = make_store()
    store.update_preferences({"MAX_TTS_CHARS": 4000, "MAX_UPLOAD_MB": 100}, base_revision=0, actor=ACTOR)
    result = store.update_preferences({}, reset=["MAX_TTS_CHARS"], base_revision=1, actor=ACTOR)
    assert result.changed == ("MAX_TTS_CHARS",) and dict(store.state.preferences.values) == {"MAX_UPLOAD_MB": 100}
    same = store.update_preferences({"MAX_UPLOAD_MB": 100.0}, base_revision=2, actor=ACTOR)
    assert same.written is False and same.revision == 2 and same.changed == ()
    document = json.loads((folder / "gateway.json").read_text(encoding="utf-8"))
    assert document["revision"] == 2 and document["values"] == {"MAX_UPLOAD_MB": 100}
    assert document["saved_by"] == {"ip": "192.168.1.20", "host": "truenas.k2o", "credential": "deployment key"}
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", document["saved_at"])


def test_values_are_checked_again_before_they_are_written(make_store, folder):
    store = make_store()
    with pytest.raises(store_module.InvalidInput) as info:
        store.update_preferences({"MAX_TTS_CHARS": 0, "BOGUS": 1, "require_key": True,
                                  "TRUSTED_HOSTS": ["*.de"]}, base_revision=0, actor=ACTOR)
    assert set(info.value.errors) == {"MAX_TTS_CHARS", "BOGUS", "require_key", "TRUSTED_HOSTS"}
    with pytest.raises(store_module.InvalidInput):
        store.update_preferences({"MAX_TTS_CHARS": 10}, reset=["MAX_TTS_CHARS"], base_revision=0, actor=ACTOR)
    with pytest.raises(store_module.InvalidInput):
        store.update_preferences({}, reset=["NOT_A_SETTING"], base_revision=0, actor=ACTOR)
    with pytest.raises(store_module.InvalidInput):
        store.update_preferences({"MAX_TTS_CHARS": 10}, base_revision="0", actor=ACTOR)
    with pytest.raises(store_module.InvalidInput) as info:
        store.update_preferences(["MAX_TTS_CHARS"], base_revision=0, actor=ACTOR)
    assert set(info.value.errors) == {"set"}
    with pytest.raises(store_module.InvalidInput) as info:
        store.update_preferences({}, reset="MAX_TTS_CHARS", base_revision=0, actor=ACTOR)
    assert set(info.value.errors) == {"reset"}
    with pytest.raises(store_module.InvalidInput) as info:
        store.update_preferences({"require_key": True}, base_revision=0, actor=ACTOR)
    assert info.value.errors == {"require_key": "This setting is not stored here."}
    assert list(folder.iterdir()) == []


def test_check_sees_the_exact_candidate_and_can_stop_the_write(make_store, folder):
    store = make_store()
    store.update_preferences({"MAX_UPLOAD_MB": 100}, base_revision=0, actor=ACTOR)
    seen = []

    class WouldLockOut(Exception):
        pass

    def refuse(values, changed):
        seen.append((dict(values), changed))
        raise WouldLockOut()

    before = (folder / "gateway.json").read_bytes()
    with pytest.raises(WouldLockOut):
        store.update_preferences({"TRUSTED_HOSTS": ["speach.k2o"]}, base_revision=1, actor=ACTOR, check=refuse)
    assert seen == [({"TRUSTED_HOSTS": ("speach.k2o",), "MAX_UPLOAD_MB": 100}, ("TRUSTED_HOSTS",))]
    assert (folder / "gateway.json").read_bytes() == before

    dry = store.update_preferences({"TRUSTED_HOSTS": ["speach.k2o"]}, base_revision=1, actor=ACTOR, dry_run=True)
    assert dry.dry_run and not dry.written and dry.revision == 2 and dry.changed == ("TRUSTED_HOSTS",)
    assert (folder / "gateway.json").read_bytes() == before
    assert store.state.preferences.revision == 1


def test_discard_and_restore(make_store, folder):
    store = make_store()
    store.update_preferences({"MAX_TTS_CHARS": 4000, "TRUSTED_HOSTS": ["speach.k2o"]}, base_revision=0, actor=ACTOR)
    discarded = store.discard(base_revision=1, actor=ACTOR)
    assert discarded.revision == 2 and set(discarded.changed) == {"MAX_TTS_CHARS", "TRUSTED_HOSTS"}
    assert dict(store.state.preferences.values) == {}
    assert json.loads((folder / "gateway.json").read_text(encoding="utf-8"))["values"] == {}
    history = store.history()
    assert [entry.event for entry in history] == ["discard", "save"]

    restored = store.restore(history[1].id, base_revision=2, actor=ACTOR)
    assert restored.revision == 3
    assert dict(store.state.preferences.values) == {"TRUSTED_HOSTS": ("speach.k2o",), "MAX_TTS_CHARS": 4000}
    assert store.history()[0].event == "restore"
    assert store.discard(base_revision=3, actor=ACTOR, dry_run=True).written is False
    assert store.state.preferences.revision == 3

    for bad in ("../../etc/passwd", "20991231T000000Z-r99", "", None, history[1].id + "/x"):
        with pytest.raises(store_module.HistoryNotFound):
            store.restore(bad, base_revision=3, actor=ACTOR)
    with pytest.raises(store_module.RevisionConflict):
        store.restore(history[1].id, base_revision=1, actor=ACTOR)

    store.discard(base_revision=3, actor=ACTOR)
    nothing = store.discard(base_revision=4, actor=ACTOR)
    assert nothing.written is False and nothing.revision == 4


def test_a_deleted_gateway_json_restarts_from_the_yaml_without_reusing_revisions(make_store, folder):
    store = make_store()
    store.update_preferences({"MAX_TTS_CHARS": 4000}, base_revision=0, actor=ACTOR)
    (folder / "gateway.json").unlink()
    assert store.refresh() is True and store.state.preferences.source == "absent"
    result = store.update_preferences({"MAX_TTS_CHARS": 3000}, base_revision=0, actor=ACTOR)
    assert result.revision == 2  # history ids stay unique
    assert len(list((folder / "history").iterdir())) == 2


# --- keys ------------------------------------------------------------------------------------

def test_keys_are_stored_as_digests_with_a_hint(make_store, folder):
    store = make_store()
    key = new_key()
    record = store.add_key(name="  Home Assistant ", role="client", key=key, actor=ACTOR)
    assert record.name == "Home Assistant" and record.role == "client"
    assert record.sha256 == hashlib.sha256(key.encode("utf-8")).hexdigest() == store_module.key_digest(key)
    assert record.hint == key[:8] and re.fullmatch(r"[0-9a-f]{8}", record.id)
    assert "sha256" not in record.public()
    text = (folder / "keys.json").read_text(encoding="utf-8")
    assert key not in text and record.sha256 in text
    keys = store.state.keys
    assert keys.revision == 1 and keys.find(key) == record and keys.find(new_key()) is None and keys.find("") is None

    with pytest.raises(store_module.InvalidInput) as info:
        store.add_key(name="home assistant", role="admin", key=new_key(), actor=ACTOR)
    assert set(info.value.errors) == {"name"}
    with pytest.raises(store_module.InvalidInput) as info:
        store.add_key(name="Other", role="admin", key=key, actor=ACTOR)
    assert set(info.value.errors) == {"key"}
    with pytest.raises(store_module.InvalidInput) as info:
        store.add_key(name="", role="root", key="tts_short", actor=ACTOR)
    assert set(info.value.errors) == {"name", "role", "key"}
    assert "tts_short" not in json.dumps(info.value.errors)


def test_at_most_32_keys(make_store):
    store = make_store()
    for index in range(32):
        store.add_key(name=f"client {index}", role="client", key=new_key(), actor=ACTOR)
    with pytest.raises(store_module.KeyLimitReached):
        store.add_key(name="one too many", role="client", key=new_key(), actor=ACTOR)


def test_the_last_admin_credential_cannot_be_revoked(make_store):
    store = make_store()
    admin = store.add_key(name="Admin", role="admin", key=new_key(), actor=ACTOR)
    client = store.add_key(name="Client", role="client", key=new_key(), actor=ACTOR)
    with pytest.raises(store_module.LastAdminCredential) as info:
        store.revoke_key(admin.id, actor=ACTOR, deployment_key_present=False)
    assert info.value.code == "last_admin_credential"
    assert store.revoke_key(client.id, actor=ACTOR, deployment_key_present=False) == client
    # With the YAML key as admin the page key may go.
    assert store.revoke_key(admin.id, actor=ACTOR, deployment_key_present=True) == admin
    with pytest.raises(store_module.KeyNotFound):
        store.revoke_key(admin.id, actor=ACTOR, deployment_key_present=True)
    assert store.state.keys.keys == ()


def test_api_access_and_the_yaml_keys_role(make_store):
    store = make_store()
    assert store.state.keys.has_admin(True) and not store.state.keys.has_admin(False)
    with pytest.raises(store_module.LastAdminCredential):
        store.set_access(base_revision=0, actor=ACTOR, deployment_key_present=True, deployment_key_role="client")
    with pytest.raises(store_module.NoKeyConfigured):
        store.set_access(base_revision=0, actor=ACTOR, deployment_key_present=False, require_key=True)
    state = store.set_access(base_revision=0, actor=ACTOR, deployment_key_present=True, require_key=True)
    assert state.require_key and state.revision == 1 and store.state.keys is state
    assert store.set_access(base_revision=1, actor=ACTOR, deployment_key_present=True, require_key=True) is state

    admin = store.add_key(name="Admin", role="admin", key=new_key(), actor=ACTOR)
    state = store.set_access(base_revision=2, actor=ACTOR, deployment_key_present=True, deployment_key_role="client")
    assert state.deployment_key_role == "client" and state.has_admin(True)
    with pytest.raises(store_module.RevisionConflict):
        store.set_access(base_revision=2, actor=ACTOR, deployment_key_present=True, require_key=False)
    # The YAML key is a client now, so the page admin is the last admin credential.
    with pytest.raises(store_module.LastAdminCredential):
        store.revoke_key(admin.id, actor=ACTOR, deployment_key_present=True)
    with pytest.raises(store_module.InvalidInput):
        store.set_access(base_revision=3, actor=ACTOR, deployment_key_present=True, deployment_key_role="root")
    with pytest.raises(store_module.InvalidInput):
        store.set_access(base_revision=3, actor=ACTOR, deployment_key_present=True, require_key="yes")
    # Whatever the file says, a YAML key means a key is required.
    open_api = store_module.KeysState(revision=0, require_key=False, deployment_key_role="admin", keys=())
    assert open_api.effective_require_key(True) and not open_api.effective_require_key(False)


# --- claiming ----------------------------------------------------------------------------------

def test_a_claim_with_the_code_creates_an_admin_key(make_store):
    store = make_store()
    code = store.issue_code(actor=ACTOR)
    with pytest.raises(store_module.InvalidCode) as info:
        store.claim(code="AAAAA-AAAAA-AAAAA-AAAAA", name="Admin", key=new_key(), actor=ACTOR)
    assert info.value.code == "invalid_code"
    key = new_key()
    record = store.claim(code=code, name="Admin", key=key, require_key=True, actor=ACTOR)
    keys = store.state.keys
    assert record.role == "admin" and keys.find(key) == record
    assert keys.require_key and keys.has_admin(False) and keys.deployment_key_role == "admin"
    with pytest.raises(store_module.InvalidCode):
        store.claim(code=code, name="Second", key=new_key(), actor=ACTOR)  # single use
    with pytest.raises(store_module.InvalidInput):
        store.claim(code=code, name="Third", key="not a key", actor=ACTOR)


def test_a_name_clash_does_not_use_up_the_code(make_store):
    store = make_store()
    store.add_key(name="Admin", role="admin", key=new_key(), actor=ACTOR)
    code = store.issue_code(actor=ACTOR)
    with pytest.raises(store_module.InvalidInput) as info:
        store.claim(code=code, name="ADMIN", key=new_key(), actor=ACTOR)
    assert set(info.value.errors) == {"name"}
    assert store.claim(code=code, name="Admin 2", key=new_key(), actor=ACTOR).role == "admin"


def test_a_claim_rewrites_a_damaged_keys_file(make_store, folder):
    (folder / "keys.json").write_text("garbage", encoding="utf-8")
    store = make_store()
    assert store.state.keys.fail_closed
    code = store.issue_code(actor=ACTOR)
    key = new_key()
    record = store.claim(code=code, name="Recovery", key=key, actor=ACTOR)
    keys = store.state.keys
    assert not keys.fail_closed and keys.keys == (record,) and keys.find(key) == record
    # Recovery does not hand the YAML key its admin role back, nor open the API.
    assert keys.require_key and keys.deployment_key_role == "client"
    assert (folder / "keys.json.damaged").read_text(encoding="utf-8") == "garbage"
    assert make_store().state.keys == keys


def test_a_full_key_list_cannot_block_recovery(make_store):
    store = make_store()
    for index in range(32):
        store.add_key(name=f"k{index}", role="admin", key=new_key(), actor=ACTOR)
    code = store.issue_code(actor=ACTOR)
    store.claim(code=code, name="Recovery", key=new_key(), actor=ACTOR)
    assert len(store.state.keys.keys) == 33


# --- audit and secrets ---------------------------------------------------------------------------

def test_every_write_is_audited_and_mirrored_to_the_log(make_store, folder, caplog):
    caplog.set_level(logging.INFO, logger=store_module.AUDIT_LOGGER_NAME)
    store = make_store()
    store.update_preferences({"TRUSTED_HOSTS": ["speach.k2o"]}, base_revision=0, actor=ACTOR)
    store.audit("refused", actor=ACTOR, result="401 invalid_api_key", path="/api/settings", key="whatever",
                code="12345", sha256="0" * 64)
    lines = [json.loads(line) for line in (folder / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
    save, refused = lines
    assert save["event"] == "save" and save["result"] == "ok" and save["revision"] == 1
    assert save["changes"] == {"TRUSTED_HOSTS": [None, ["speach.k2o"]]}
    assert (save["ip"], save["host"], save["credential"]) == ("192.168.1.20", "truenas.k2o", "deployment key")
    assert save["worker"] == os.getpid() and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", save["time"])
    assert refused["event"] == "refused" and refused["path"] == "/api/settings"
    assert not {"key", "code", "sha256"} & set(refused)
    mirrored = [record.getMessage() for record in caplog.records if record.name == store_module.AUDIT_LOGGER_NAME]
    assert [json.loads(message)["event"] for message in mirrored] == ["save", "refused"]


def test_the_audit_log_rotates_at_its_size_limit(make_store, folder, monkeypatch):
    monkeypatch.setattr(store_module, "AUDIT_MAX_BYTES", 1000)
    store = make_store()
    for attempt in range(80):
        store.audit("refused", actor=ACTOR, result=f"attempt {attempt}")
    names = sorted(path.name for path in folder.iterdir() if path.name.startswith("audit"))
    assert names == ["audit.jsonl", "audit.jsonl.1", "audit.jsonl.2", "audit.jsonl.3"]
    assert all((folder / name).stat().st_size <= 1000 for name in names)
    newest = json.loads((folder / "audit.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert newest["result"] == "attempt 79"


def test_without_the_mount_the_audit_goes_to_the_log_only(tmp_path, lock_path, caplog):
    caplog.set_level(logging.INFO, logger=store_module.AUDIT_LOGGER_NAME)
    missing = tmp_path / "not-mounted"
    SettingsStore(missing, verify_mount=False, lock_path=lock_path).audit("refused", actor=ACTOR)
    assert any(record.name == store_module.AUDIT_LOGGER_NAME for record in caplog.records)
    assert not missing.exists()


def test_no_file_or_log_line_holds_a_plaintext_key_or_code(make_store, folder, caplog):
    caplog.set_level(logging.DEBUG)
    store = make_store()
    client_key, admin_key = new_key(), new_key()
    store.add_key(name="Client", role="client", key=client_key, actor=ACTOR)
    code = store.issue_code(actor=ACTOR)
    store.redeem_code("ZZZZZ-ZZZZZ-ZZZZZ-ZZZZZ", actor=ACTOR)
    store.claim(code=code, name="Admin", key=admin_key, actor=ACTOR)
    # A careless caller cannot leak one through the audit either.
    store.audit("note", actor=ACTOR, detail=f"got {admin_key} and {code}", nested={"list": [client_key]})
    on_disk = everything_in(folder)
    logged = caplog.text
    for secret in (client_key, admin_key, code, code.replace("-", "")):
        assert secret not in on_disk and secret not in logged
    audit = (folder / "audit.jsonl").read_text(encoding="utf-8")
    for record in store.state.keys.keys:
        assert record.sha256 not in audit


# --- a write asks again whether its credential may write -------------------------------------------

def _revoked_meanwhile(key_id):
    """An `authorize` that refuses once the page key `key_id` is gone from the keys just read."""
    def authorize(keys):
        if not any(record.id == key_id for record in keys.keys):
            raise store_module.CredentialRefused("invalid_api_key", "revoked")
    return authorize


def test_every_write_asks_authorize_under_the_lock_with_the_keys_just_read(make_store, folder):
    """The key that let a request in may be revoked by another worker while the request waits for its
    body: every write judges it again, against keys.json as read under the lock, and writes nothing."""
    owner_store, parked = make_store(), make_store()
    owner = owner_store.add_key(name="Owner", role="admin", key=new_key(), actor=ACTOR)
    stolen = owner_store.add_key(name="Stolen", role="admin", key=new_key(), actor=ACTOR)
    owner_store.update_preferences({"MAX_TTS_CHARS": 4000}, base_revision=0, actor=ACTOR)
    parked.refresh()
    assert {record.id for record in parked.state.keys.keys} == {owner.id, stolen.id}
    seen = []

    def spying(keys):
        seen.append(tuple(record.id for record in keys.keys))
        _revoked_meanwhile(stolen.id)(keys)

    owner_store.revoke_key(stolen.id, actor=ACTOR, deployment_key_present=False)       # in the other worker
    before = {name: (folder / name).read_bytes() for name in ("gateway.json", "keys.json")}
    history = parked.history()[0].id
    attempts = [
        lambda: parked.update_preferences({"MAX_TTS_CHARS": 1}, base_revision=1, actor=ACTOR, authorize=spying),
        lambda: parked.update_preferences({"MAX_TTS_CHARS": 1}, base_revision=1, actor=ACTOR, dry_run=True,
                                          authorize=spying),
        lambda: parked.discard(base_revision=1, actor=ACTOR, authorize=spying),
        lambda: parked.restore(history, base_revision=1, actor=ACTOR, authorize=spying),
        lambda: parked.confirm(1, actor=ACTOR, authorize=spying),
        lambda: parked.set_access(base_revision=3, actor=ACTOR, deployment_key_present=False, require_key=True,
                                  authorize=spying),
        lambda: parked.add_key(name="Backdoor", role="admin", key=new_key(), actor=ACTOR, authorize=spying),
        lambda: parked.revoke_key(owner.id, actor=ACTOR, deployment_key_present=False, authorize=spying),
    ]
    for attempt in attempts:
        with pytest.raises(store_module.CredentialRefused) as info:
            attempt()
        assert info.value.code == "invalid_api_key"
    assert seen == [(owner.id,)] * len(attempts), "authorize did not see keys.json as it is now"
    assert {name: (folder / name).read_bytes() for name in before} == before
    # A credential that still may write goes through.
    parked.add_key(name="Phone", role="client", key=new_key(), actor=ACTOR, authorize=_revoked_meanwhile(owner.id))
    assert [record.name for record in make_store().state.keys.keys] == ["Owner", "Phone"]


# --- a read that fails for a reason that may pass decides nothing --------------------------------------

def _failing_once(monkeypatch, store, name, error=errno.EMFILE):
    """`store`'s next read of the file `name` fails with `error` (out of file descriptors), once."""
    real = store._read_bytes
    left = {"count": 1}

    def read(path):
        if Path(path).name == name and left["count"]:
            left["count"] -= 1
            raise OSError(error, os.strerror(error))
        return real(path)

    monkeypatch.setattr(store, "_read_bytes", read)
    return left


def test_a_gateway_json_read_that_may_pass_keeps_what_is_in_force_and_is_tried_again(make_store, monkeypatch):
    monkeypatch.setattr(store_module, "READ_RETRY_S", 0.0, raising=False)
    writer, worker = make_store(), make_store()
    writer.update_preferences({"TRUSTED_HOSTS": ["old.example.com"]}, base_revision=0, actor=ACTOR)
    worker.refresh()
    writer.update_preferences({}, reset=["TRUSTED_HOSTS"], base_revision=1, actor=ACTOR)   # a tightening
    _failing_once(monkeypatch, worker, "gateway.json")
    worker.refresh()
    preferences = worker.state.preferences
    assert preferences.revision == 1 and preferences.values["TRUSTED_HOSTS"] == ("old.example.com",)
    assert not preferences.damaged, "a read that may pass is no verdict on the file"
    reads = worker.reads
    assert worker.refresh() is True, "the file was not read again"
    assert worker.reads == reads + 1
    assert worker.state.preferences.revision == 2 and "TRUSTED_HOSTS" not in worker.state.preferences.values
    assert worker.refresh() is False and worker.reads == reads + 1        # read once it worked


def test_a_keys_json_read_that_may_pass_fails_closed_only_until_it_can_be_read(make_store, monkeypatch):
    monkeypatch.setattr(store_module, "READ_RETRY_S", 0.0, raising=False)
    writer, worker = make_store(), make_store()
    home = new_key()
    writer.add_key(name="Home", role="admin", key=home, actor=ACTOR)
    worker.refresh()
    writer.add_key(name="Phone", role="client", key=new_key(), actor=ACTOR)
    _failing_once(monkeypatch, worker, "keys.json")
    worker.refresh()
    keys = worker.state.keys
    assert keys.fail_closed and keys.find(home) is None, "the change it could not read may be a revocation"
    assert "could not be read" in keys.problem and "EMFILE" in keys.problem
    assert worker.refresh() is True, "the file was not read again"
    keys = worker.state.keys
    assert not keys.fail_closed and keys.find(home).name == "Home" and len(keys.keys) == 2


def test_a_save_on_a_worker_that_missed_a_read_is_a_conflict_not_a_revert(make_store, folder, monkeypatch):
    """The write reads the file again under the lock: the stale worker's revision is refused, and the
    other worker's change (here: a host name removed) is never written back over."""
    monkeypatch.setattr(store_module, "READ_RETRY_S", 3600.0, raising=False)
    writer, worker = make_store(), make_store()
    writer.update_preferences({"TRUSTED_HOSTS": ["old.example.com"]}, base_revision=0, actor=ACTOR)
    worker.refresh()
    writer.update_preferences({}, reset=["TRUSTED_HOSTS"], base_revision=1, actor=ACTOR)
    _failing_once(monkeypatch, worker, "gateway.json")
    worker.refresh()
    assert worker.state.preferences.revision == 1                  # what this worker shows meanwhile
    with pytest.raises(store_module.RevisionConflict) as info:
        worker.update_preferences({"MAX_TTS_CHARS": 4000}, base_revision=1, actor=ACTOR)
    assert info.value.current_revision == 2
    document = json.loads((folder / "gateway.json").read_text(encoding="utf-8"))
    assert document["revision"] == 2 and "TRUSTED_HOSTS" not in document["values"]
    assert not (folder / "gateway.json.damaged").exists()


def test_a_write_whose_own_read_fails_is_refused_and_writes_nothing(make_store, folder, monkeypatch):
    store = make_store()
    store.update_preferences({"MAX_TTS_CHARS": 4000}, base_revision=0, actor=ACTOR)
    before = (folder / "gateway.json").read_bytes()
    _failing_once(monkeypatch, store, "gateway.json")
    with pytest.raises(store_module.SettingsStoreError) as info:
        store.update_preferences({"MAX_TTS_CHARS": 3000}, base_revision=1, actor=ACTOR)
    assert isinstance(info.value, store_module.Unreadable) and info.value.code == "settings_unreadable"
    assert (folder / "gateway.json").read_bytes() == before
    assert store.update_preferences({"MAX_TTS_CHARS": 3000}, base_revision=1, actor=ACTOR).revision == 2


def test_a_claim_on_a_worker_that_missed_a_read_keeps_every_key(make_store, folder, monkeypatch):
    """Damage is decided by the read under the lock, never by what a worker cached: a healthy
    keys.json one worker failed to read is not rewritten as if it were damaged."""
    monkeypatch.setattr(store_module, "READ_RETRY_S", 3600.0, raising=False)
    writer, worker = make_store(), make_store()
    writer.add_key(name="Home", role="client", key=new_key(), actor=ACTOR)
    worker.refresh()
    writer.add_key(name="Laptop", role="admin", key=new_key(), actor=ACTOR)
    writer.set_access(base_revision=2, actor=ACTOR, deployment_key_present=True, require_key=True)
    _failing_once(monkeypatch, worker, "keys.json")
    worker.refresh()
    assert worker.state.keys.fail_closed                       # what this worker believes meanwhile
    code = worker.issue_code(actor=ACTOR)
    worker.claim(code=code, name="Recovered", key=new_key(), actor=ACTOR)
    keys = make_store().state.keys
    assert [record.name for record in keys.keys] == ["Home", "Laptop", "Recovered"]
    assert keys.require_key and keys.deployment_key_role == "admin" and not keys.fail_closed
    assert not (folder / "keys.json.damaged").exists()


@POSIX_ONLY
def test_a_keys_file_it_may_not_read_fails_closed_until_its_mode_is_repaired(make_store, folder, monkeypatch):
    if os.geteuid() == 0:
        pytest.skip("root reads the file whatever its mode")
    writer, worker = make_store(), make_store()
    key = new_key()
    writer.add_key(name="Home", role="admin", key=key, actor=ACTOR)
    path = folder / "keys.json"
    path.chmod(0)
    tried = []
    real = worker._read_bytes
    monkeypatch.setattr(worker, "_read_bytes", lambda name: (tried.append(Path(name).name), real(name))[1])
    try:
        keys = worker.state.keys
        assert keys.fail_closed and "PermissionError" in keys.problem
        assert worker.refresh() is False and tried.count("keys.json") == 1, "a verdict is kept until the file changes"
    finally:
        path.chmod(0o600)
    assert worker.refresh() is True, "chmod is a change of the file (its mode and ctime)"
    assert worker.state.keys.find(key).name == "Home"


@POSIX_ONLY
def test_a_recovery_keeps_a_keys_file_it_may_not_read_as_it_is(make_store, folder):
    if os.geteuid() == 0:
        pytest.skip("root reads the file whatever its mode")
    make_store().add_key(name="Home", role="admin", key=new_key(), actor=ACTOR)
    path = folder / "keys.json"
    inode = path.stat().st_ino
    path.chmod(0)
    store = make_store()
    assert store.state.keys.fail_closed
    store.claim(code=store.issue_code(actor=ACTOR), name="Recovered", key=new_key(), actor=ACTOR)
    kept = folder / "keys.json.damaged"
    assert kept.stat().st_ino == inode and temp_files(folder) == []
    kept.chmod(0o600)
    assert json.loads(kept.read_text(encoding="utf-8"))["keys"][0]["name"] == "Home"
    assert [record.name for record in store.state.keys.keys] == ["Recovered"]


# --- names and ids of credentials --------------------------------------------------------------------

def test_a_name_the_audit_trail_uses_is_refused_for_a_new_key_but_read_from_an_old_file(make_store, folder):
    store = make_store()
    code = store.issue_code(actor=ACTOR)
    codes_before = (folder / "codes.json").read_bytes()
    for name in ("deployment key", "Deployment Key", " DEPLOYMENT   KEY ", "one-time code", "AUTO-REVERT"):
        with pytest.raises(store_module.InvalidInput) as info:
            store.add_key(name=name, role="admin", key=new_key(), actor=ACTOR)
        assert set(info.value.errors) == {"name"} and "reserved" in info.value.errors["name"], name
        with pytest.raises(store_module.InvalidInput):
            store.claim(code=code, name=name, key=new_key(), actor=ACTOR)
    assert not (folder / "keys.json").exists()
    assert (folder / "codes.json").read_bytes() == codes_before, "a refused name cost the code something"
    assert store.claim(code=code, name="Deployment helper", key=new_key(), actor=ACTOR).role == "admin"

    # A keys.json that already holds such a name (an earlier version let it be made) is read as it is.
    write_json(folder / "keys.json", _valid_keys_document(keys=[
        {"id": "9f2c1a7b", "name": "deployment key", "role": "admin", "sha256": "a" * 64, "hint": "tts_Ab3d",
         "created_at": "2026-10-02T09:12:03Z"}]))
    keys = make_store().state.keys
    assert not keys.fail_closed and keys.keys[0].name == "deployment key"


def test_who_saved_is_recorded_by_name_and_by_credential_id(make_store, folder):
    store = make_store()
    by_yaml = Actor(ip="192.168.1.20", host="truenas.k2o", credential=store_module.DEPLOYMENT_KEY,
                    credential_id=store_module.DEPLOYMENT_KEY_ID)
    store.update_preferences({"MAX_TTS_CHARS": 4000}, base_revision=0, actor=by_yaml)
    document = json.loads((folder / "gateway.json").read_text(encoding="utf-8"))
    assert document["saved_by"] == {"ip": "192.168.1.20", "host": "truenas.k2o", "credential": "deployment key",
                                    "credential_id": "yaml:API_KEY"}
    assert make_store().state.preferences.saved_by["credential_id"] == "yaml:API_KEY"
    assert store.history()[0].saved_by["credential_id"] == "yaml:API_KEY"
    assert json.loads((folder / "audit.jsonl").read_text(encoding="utf-8").splitlines()[-1])["credential_id"] == \
        "yaml:API_KEY"
    # The id of the key a line is about stays its own field next to the credential's.
    record = store.add_key(name="Phone", role="client", key=new_key(),
                           actor=Actor(credential="Laptop", credential_id="0a1b2c3d"))
    line = json.loads((folder / "audit.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert (line["key_id"], line["credential_id"], line["credential"]) == (record.id, "0a1b2c3d", "Laptop")


def test_a_claim_on_a_claimed_server_is_audited_as_a_recovery(make_store, folder, clock):
    store = make_store()

    def claim_line(name, **kwargs):
        code = store.issue_code(actor=ACTOR)
        clock.advance(61)
        store.claim(code=code, name=name, key=new_key(), actor=ACTOR, **kwargs)
        line = json.loads((folder / "audit.jsonl").read_text(encoding="utf-8").splitlines()[-1])
        return line["event"], line["server_was_claimed"], line["recovered"]

    assert claim_line("Owner") == ("claim", False, False)
    assert claim_line("Lost my key") == ("recovery", True, False)
    (folder / "keys.json").write_text("garbage", encoding="utf-8")
    assert claim_line("After the damage") == ("recovery", False, True)
    # The gateway says what counts as claimed (the YAML key as admin, say), asked under the lock.
    asked = []
    assert claim_line("Asked", claimed=lambda keys: asked.append(keys) or True) == ("recovery", True, False)
    assert len(asked) == 1 and [record.name for record in asked[0].keys] == ["After the damage"]


def test_a_refusal_for_the_log_only_takes_no_lock_and_touches_no_file(make_store, folder, caplog, monkeypatch):
    caplog.set_level(logging.INFO, logger=store_module.AUDIT_LOGGER_NAME)
    store = make_store()
    taken = []
    real_acquire = store_module.FileLock.acquire

    def acquire(lock):
        taken.append(lock.path)
        return real_acquire(lock)

    monkeypatch.setattr(store_module.FileLock, "acquire", acquire)
    store.audit("refused", actor=ACTOR, result="403 cross_origin_blocked", to_file=False, path="/api/settings")
    assert taken == [] and list(folder.iterdir()) == []
    lines = [json.loads(record.getMessage()) for record in caplog.records
             if record.name == store_module.AUDIT_LOGGER_NAME]
    assert [(line["event"], line["result"], line["path"]) for line in lines] == [
        ("refused", "403 cross_origin_blocked", "/api/settings")]
    assert "to_file" not in lines[0]
