"""The files behind the Settings page: reading, locking and atomic writes (stdlib only).

The folder is the bind mount /app/settings. No image creates it and neither does this
module: it only ever writes into a folder that exists and is verified (below).

    gateway.json               the preference overrides changed in the page, never a snapshot:
                               {"schema", "revision", "saved_at", "saved_by", "values", "pending"}
    keys.json                  the access state: {"schema", "revision", "require_key",
                               "deployment_key_role", "keys": [{"id", "name", "role", "sha256",
                               "hint", "created_at"}]}
    codes.json                 one-time claim codes, as SHA-256 only
    history/<UTC>-r<rev>.json  the last 20 versions of gateway.json, each with the event, the
                               changed key names and who saved it (made on the first save, 0700)
    audit.jsonl, .1 to .3      one JSON line per save, claim, key change, automatic undo, and per
                               refusal of a presented key or code or of an admin's change, 1 MiB per
                               file; every line also goes to the logger "tts_stt.settings.audit"
                               (the container log, which the app cannot edit). Refusals anyone can
                               cause without an admin key go to that logger only (audit(to_file=False)),
                               so they can never push this record out of its four files.
    SAFE-MODE                  made by hand; the gateway then ignores gateway.json
    *.damaged                  the last damaged gateway.json / keys.json a write replaced

Guarantees
  - Importing the module or constructing a store reads nothing and writes nothing. Reads start
    with the first refresh(); the first write creates files, history/ and the lock file.
  - The default folder counts as writable only when /proc/self/mountinfo lists it as a
    read-write mount point. TTS_STT_SETTINGS_DIR (tests, local runs) names another folder and
    skips that check; it must exist.
  - Every write holds an exclusive lock on /tmp/tts-stt-settings.lock (container-local and shared
    by the uvicorn workers; fcntl.flock, msvcrt on Windows, a thread lock where neither exists),
    re-reads gateway.json and keys.json whatever this process cached (a read that fails for a
    reason that may pass refuses the write: Unreadable), asks the caller's `authorize` whether the
    credential that let the request in still may write, checks the revision the caller based the
    change on, writes a temp file in the same folder (mode 0600), fsyncs it, os.replace()s it over
    the target and fsyncs the folder (best effort).
  - gateway.json is read leniently: an invalid value is dropped with a problem message, unknown
    keys are ignored, and a damaged file leaves the last good state of this process in force; a
    process that never had one takes the newest valid history version, else no overrides.
  - keys.json fails closed: damaged, or unreadable for want of permission, means a key is required,
    there are no page keys and the app YAML key has the role client, until claim() rewrites it or it
    is deleted.
  - A read that fails for a reason that may pass (out of file descriptors, an I/O error) decides
    nothing and is not remembered: keys.json counts as unreadable (fail closed) and gateway.json
    keeps what was in force until a read succeeds, tried again on the next refresh after
    READ_RETRY_S. Only a read with a definite result (the content, or a verdict on it) is cached.
  - Keys and codes are stored as SHA-256 digests only, and no plaintext key or code is ever
    written to a file, the audit trail or a log line by this module.

Use from the gateway

    store = SettingsStore.from_environment()     # /app/settings, or TTS_STT_SETTINGS_DIR
    # once per HTTP/WebSocket request, before the guard:
    if store.refresh():                           # stats 4 paths; re-reads only a changed file
        apply(store.state)                        # SettingsState: mount, safe_mode, preferences, keys
    if store.expire_pending():                    # undo an unconfirmed commit-confirm change
        apply(store.state)

State objects are immutable and replaced as a whole, so `new.preferences is old.preferences`
tells whether gateway.json changed (the same for keys and mount).

  SettingsState(mount: MountStatus, safe_mode: bool, preferences: PreferencesState, keys: KeysState)
  MountStatus(state: "mounted" | "override" | "not_mounted" | "read_only" | "missing",
              writable: bool, directory: str, detail: str)
  Actor(ip, host, credential, credential_id)  who asks; credential_id is a page key's id,
              DEPLOYMENT_KEY_ID for the app YAML key, "" otherwise (never part of a name)
  PreferencesState(revision, values {KEY: normalized value}, pending: Pending | None,
              saved_at, saved_by {ip, host, credential[, credential_id]}, source: "absent" | "file" |
              "last_good" | "history" | "none", damaged: bool, problems: tuple[str],
              dropped {KEY: reason})
  Pending(revision, deadline (epoch s), previous {KEY: override before, None = followed the YAML})
  KeysState(revision, require_key, deployment_key_role, keys: tuple[KeyRecord], fail_closed,
              problem) with .find(presented_key) -> KeyRecord | None, .has_admin(deployment_key_present),
              .effective_require_key(deployment_key_present), .public()
  KeyRecord(id, name, role, sha256, hint, created_at) with .public() (no digest)
  HistoryEntry(id, revision, saved_at, saved_by, event, changed) with .as_dict()
  WriteResult(revision, changed: tuple[str], values, pending, dropped, dry_run, written)
      A write that created or moved a confirmation answers pending.revision == revision.

Writes (all raise NotWritable when the folder is not writable, LockTimeout, Unreadable, WriteFailed
on I/O). Those an admin credential asks for take `authorize`: called under the lock with the
KeysState just read, it raises (CredentialRefused) when the key that let the request in has been
revoked or made a client since, and nothing is written.
  update_preferences(set_values, reset=(), *, base_revision, actor, check=None, dry_run=False, authorize=None)
  restore(history_id, *, base_revision, actor, check=None, dry_run=False, authorize=None)
  discard(*, base_revision, actor, check=None, dry_run=False, authorize=None)
      -> WriteResult; RevisionConflict, InvalidInput (values are re-checked with
      settings_schema.parse_value), HistoryNotFound. `check(values, changed)` runs under the lock
      with the exact override map about to be written and may raise to stop the write (the
      gateway's lockout self-check); it must not do I/O. A change to a commit_confirm key gets a
      Pending record (deadline now + 60 s), and later writes carry it over.
  confirm(revision, *, actor=None, authorize=None) -> bool   clears the pending record of that revision
  expire_pending() -> bool                        never raises
  set_access(*, base_revision, actor, deployment_key_present, require_key=None,
             deployment_key_role=None, authorize=None) -> KeysState
      RevisionConflict, InvalidInput, LastAdminCredential, NoKeyConfigured, KeysFileDamaged
  add_key(*, name, role, key, actor, authorize=None) -> KeyRecord
      InvalidInput (also for a name the audit trail reserves), KeyLimitReached (32), KeysFileDamaged
  revoke_key(key_id, *, actor, deployment_key_present, authorize=None) -> KeyRecord
      KeyNotFound, LastAdminCredential, KeysFileDamaged
  issue_code(*, purpose="claim", actor=None) -> str
      The only copy of the code: the caller prints it to the container log and nowhere else.
      RateLimited (one per minute, at most 3 active)
  redeem_code(code, *, purpose="claim", actor=None) -> bool
      A wrong code changes nothing: no attempt budget is shared between callers, so nobody can
      use up the owner's code. Guessing is bounded by 100 bits, 30 minutes, at most 3 codes and the
      gateway's per-address limit on wrong codes.
  claim(*, code, name, key, require_key=None, actor, claimed=None) -> KeyRecord
      Creates an admin key with a one-time code (also on a damaged keys.json, which it rewrites).
      Audited as "claim", or as "recovery" when the server already had an admin credential
      (`claimed(keys)`) or keys.json was damaged. InvalidInput, InvalidCode
  audit(event, *, actor=None, result="ok", to_file=True, **fields) -> None     never raises

Reads: refresh() -> bool, state, mount_status(), history(limit=20) -> list[HistoryEntry],
reads (a counter of file reads, for tests). Module helpers: key_digest(key),
parse_mountinfo(text), find_mount(entries, path), FileLock.

Every error is a SettingsStoreError with a stable `code` and a `message` safe to show.
"""

from __future__ import annotations

import errno
import functools
import hashlib
import hmac
import json
import logging
import math
import os
import posixpath
import re
import secrets
import stat
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Optional

import settings_schema as schema

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None
try:
    import msvcrt
except ImportError:  # POSIX
    msvcrt = None

logger = logging.getLogger("tts_stt.settings")
AUDIT_LOGGER_NAME = "tts_stt.settings.audit"
audit_logger = logging.getLogger(AUDIT_LOGGER_NAME)

DEFAULT_DIRECTORY = "/app/settings"
OVERRIDE_ENV = "TTS_STT_SETTINGS_DIR"
DEFAULT_LOCK_PATH = "/tmp/tts-stt-settings.lock"
MOUNTINFO_PATH = "/proc/self/mountinfo"

PREFERENCES_FILE = "gateway.json"
KEYS_FILE = "keys.json"
CODES_FILE = "codes.json"
HISTORY_DIR = "history"
AUDIT_FILE = "audit.jsonl"
SAFE_MODE_FILE = "SAFE-MODE"

FILE_SCHEMA = 1
HISTORY_LIMIT = 20
AUDIT_MAX_BYTES = 1024 * 1024
AUDIT_BACKUPS = 3
MAX_KEYS = 32
MAX_FILE_BYTES = 1024 * 1024
CONFIRM_WINDOW_S = 60.0
LOCK_TIMEOUT_S = 10.0
# How soon a refresh tries a file again whose read failed for a reason that may pass.
READ_RETRY_S = 1.0

CODE_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"   # Crockford base32: 5 bits a character
CODE_LENGTH = 20                                      # 100 bits
CODE_TTL_S = 30 * 60
CODE_MAX_ACTIVE = 3
CODE_MIN_INTERVAL_S = 60.0
CODE_PURPOSES = ("claim",)

# The names audit lines and the history give the credentials that are not page keys: the app
# YAML's API_KEY, a claim with a one-time code, and an unconfirmed change undone by its deadline.
# A page key cannot be named like one of them (see _checked_key_fields).
DEPLOYMENT_KEY = "deployment key"
CLAIM_ACTOR = "one-time code"
AUTO_REVERT_ACTOR = "auto-revert"
RESERVED_CREDENTIAL_NAMES = frozenset({DEPLOYMENT_KEY, CLAIM_ACTOR, AUTO_REVERT_ACTOR})
# Actor.credential_id of the app YAML's API_KEY. The colon keeps it apart from every page key id
# (letters, digits, "_" and "-"), so the record says whose key it was, whatever the key's name.
DEPLOYMENT_KEY_ID = "yaml:API_KEY"

# A read that failed with one of these is a verdict on the file, as damaged content is: it stays
# in force until the file changes (its owner, mode or ctime included). Any other error (out of
# file descriptors or memory, an I/O error) may pass, and the file is read again later.
_PERSISTENT_READ_ERRORS = frozenset({errno.EACCES, errno.EPERM})

MOUNT_MOUNTED = "mounted"
MOUNT_OVERRIDE = "override"
MOUNT_NOT_MOUNTED = "not_mounted"
MOUNT_READ_ONLY = "read_only"
MOUNT_MISSING = "missing"

SOURCE_ABSENT = "absent"        # no gateway.json: the app YAML applies
SOURCE_FILE = "file"
SOURCE_LAST_GOOD = "last_good"  # damaged file; this process's last good values stay
SOURCE_HISTORY = "history"      # damaged file at start; the newest valid history version
SOURCE_NONE = "none"            # damaged file and nothing to fall back to: the app YAML applies


# --- errors ---------------------------------------------------------------------------------

class SettingsStoreError(Exception):
    """Base of every refusal; `code` is stable, `message` is safe to show to the caller."""

    code = "settings_error"
    default_message = "The settings could not be changed."

    def __init__(self, message: Optional[str] = None):
        self.message = message or self.default_message
        super().__init__(self.message)


class NotWritable(SettingsStoreError):
    code = "settings_not_mounted"
    default_message = "Settings cannot be saved: the settings folder is not mounted from the app's dataset."


class WriteFailed(SettingsStoreError):
    code = "settings_write_failed"
    default_message = "The settings could not be written."


class LockTimeout(SettingsStoreError):
    code = "settings_busy"
    default_message = "The settings are being changed by another request; try again."


class Unreadable(SettingsStoreError):
    """A file a write depends on could not be read just now, for a reason that may pass."""

    code = "settings_unreadable"
    default_message = "The settings files could not be read just now; try again."


class CredentialRefused(SettingsStoreError):
    """Raised by a write's `authorize`: the key that let the request in may not write any more.

    `code` is the refusal the request would get if it came now: "invalid_api_key" for a key
    that is gone (revoked), "admin_key_required" for one that is no longer an admin.
    """

    code = "invalid_api_key"
    default_message = "This key is no longer accepted; nothing was changed."

    def __init__(self, code: str = "invalid_api_key", message: Optional[str] = None):
        super().__init__(message)
        self.code = code


class RevisionConflict(SettingsStoreError):
    code = "revision_conflict"
    default_message = "The settings were changed in the meantime; reload and try again."

    def __init__(self, current_revision: int, message: Optional[str] = None):
        super().__init__(message)
        self.current_revision = current_revision


class InvalidInput(SettingsStoreError):
    code = "invalid_input"
    default_message = "Some values are not valid."

    def __init__(self, errors: Mapping[str, str], message: Optional[str] = None):
        super().__init__(message)
        self.errors = dict(errors)


class HistoryNotFound(SettingsStoreError):
    code = "history_not_found"
    default_message = "That saved version does not exist (any more)."


class KeyNotFound(SettingsStoreError):
    code = "key_not_found"
    default_message = "There is no such key."


class KeyLimitReached(SettingsStoreError):
    code = "key_limit_reached"
    default_message = f"At most {MAX_KEYS} keys; revoke one first."


class LastAdminCredential(SettingsStoreError):
    code = "last_admin_credential"
    default_message = "This is the last credential that may change settings; create another admin key first."


class NoKeyConfigured(SettingsStoreError):
    code = "no_key_configured"
    default_message = "Requiring a key needs a key that works: create one first."


class KeysFileDamaged(SettingsStoreError):
    code = "keys_file_damaged"
    default_message = ("keys.json is damaged, so only the app YAML key is accepted. Print a recovery "
                       "code and claim the server again to rewrite it.")


class InvalidCode(SettingsStoreError):
    code = "invalid_code"
    default_message = "The code is wrong, used or expired."


class RateLimited(SettingsStoreError):
    code = "rate_limited"
    default_message = "Too many codes; try again later."

    def __init__(self, retry_after: int, message: Optional[str] = None):
        super().__init__(message)
        self.retry_after = max(1, int(retry_after))


class _Damaged(Exception):
    """A file exists but cannot be used; the text says why (no file content in it)."""


# --- small helpers --------------------------------------------------------------------------

def key_digest(key: str) -> str:
    """The SHA-256 hex digest a key is stored and compared as."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _stamp(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _clip(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else ("" if value is None else str(value))
    return "".join(ch if ch.isprintable() else "?" for ch in text[:limit])


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _jsonable(value: Any) -> Any:
    return list(value) if isinstance(value, tuple) else value


def _dump(document: Mapping) -> bytes:
    return (json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _reason(exc: BaseException) -> str:
    if isinstance(exc, _Damaged):
        return str(exc)
    if isinstance(exc, schema.DuplicateKeyError):
        return "a key appears twice"
    if isinstance(exc, ValueError):
        return "not valid JSON"
    if isinstance(exc, OSError) and exc.errno in errno.errorcode:
        return f"{type(exc).__name__} {errno.errorcode[exc.errno]}"
    return type(exc).__name__


def _may_pass(exc: OSError) -> bool:
    """Did a read fail for a reason that may pass (descriptors, memory, I/O), not a verdict?"""
    return exc.errno not in _PERSISTENT_READ_ERRORS


def _reserved_name(name: str) -> bool:
    """Is `name` (case and inner spaces aside) one the audit trail uses for another credential?"""
    folded = " ".join(name.split()).casefold()
    return any(folded == reserved.casefold() for reserved in RESERVED_CREDENTIAL_NAMES)


def _fsync_directory(directory: str) -> None:
    if os.name != "posix":
        return
    try:
        fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


_SECRET_FIELDS = frozenset({"key", "code", "sha256", "secret", "token", "password", "api_key",
                            "authorization"})
_REDACTIONS = (
    (re.compile(r"tts_[A-Za-z0-9_-]{43}"), "[key]"),
    (re.compile(r"\b[0-9A-Za-z]{5}(?:-[0-9A-Za-z]{5}){3}\b"), "[code]"),
    (re.compile(r"\b[0-9a-fA-F]{64}\b"), "[digest]"),
)


def _redact(text: str) -> str:
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


def _scrub(value: Any, depth: int = 0) -> Any:
    """A copy safe for the audit trail: no secret-named fields, nothing that looks like a key."""
    if depth > 6:
        return "..."
    if isinstance(value, Mapping):
        return {_clip(k, 64): _scrub(v, depth + 1) for k, v in value.items()
                if str(k).lower() not in _SECRET_FIELDS}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_scrub(item, depth + 1) for item in list(value)[:100]]
    if isinstance(value, bool) or value is None or _is_int(value):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return _redact(_clip(value, 2000))


# --- /proc/self/mountinfo -------------------------------------------------------------------

@dataclass(frozen=True)
class MountEntry:
    mount_id: int
    parent_id: int
    root: str
    mount_point: str
    options: frozenset
    fs_type: str
    source: str

    @property
    def read_only(self) -> bool:
        return "ro" in self.options


_MOUNTINFO_ESCAPE = re.compile(r"\\([0-7]{3})")


def _unescape(field: str) -> str:
    return _MOUNTINFO_ESCAPE.sub(lambda match: chr(int(match.group(1), 8)), field)


def parse_mountinfo(text: str) -> list[MountEntry]:
    """The entries of a /proc/<pid>/mountinfo text; malformed lines are skipped.

    Format (proc(5)): id parent major:minor root mount-point options [optional...] - fstype
    source super-options, with space, tab, newline and backslash escaped as \\ooo.
    """
    entries: list[MountEntry] = []
    for line in text.splitlines():
        fields = line.split(" ")
        if len(fields) < 7:
            continue
        try:
            separator = fields.index("-", 6)
            mount_id, parent_id = int(fields[0]), int(fields[1])
        except ValueError:
            continue
        tail = fields[separator + 1:]
        entries.append(MountEntry(
            mount_id=mount_id, parent_id=parent_id, root=_unescape(fields[3]),
            mount_point=_unescape(fields[4]), options=frozenset(fields[5].split(",")),
            fs_type=_unescape(tail[0]) if tail else "",
            source=_unescape(tail[1]) if len(tail) > 1 else ""))
    return entries


def find_mount(entries: Iterable[MountEntry], path: str) -> Optional[MountEntry]:
    """The entry mounted exactly at `path` (the last one when several are stacked), or None."""
    target = posixpath.normpath(path)
    found = None
    for entry in entries:
        if posixpath.normpath(entry.mount_point) == target:
            found = entry
    return found


# --- the cross-process lock -----------------------------------------------------------------

def _try_lock(fd: int) -> bool:
    if fcntl is not None:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
                return False
            raise
        return True
    if msvcrt is not None:
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True
    return True


def _unlock(fd: int) -> None:
    try:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_UN)
        elif msvcrt is not None:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    except OSError:
        pass  # closing the descriptor releases it anyway


class FileLock:
    """An exclusive lock on a small file, shared by every process that names the same path.

    Re-entrant within a thread, exclusive between threads and between processes. Nothing is
    created until the first acquire().
    """

    def __init__(self, path: Optional[str] = None, *, timeout: float = LOCK_TIMEOUT_S):
        self._path = os.fspath(path) if path is not None else None
        self._timeout = timeout
        self._thread_lock = threading.RLock()
        self._depth = 0
        self._fd: Optional[int] = None

    @property
    def path(self) -> str:
        if self._path is None:
            if os.name == "posix":
                self._path = DEFAULT_LOCK_PATH
            else:
                self._path = os.path.join(tempfile.gettempdir(), "tts-stt-settings.lock")
        return self._path

    def acquire(self) -> None:
        deadline = time.monotonic() + self._timeout
        if not self._thread_lock.acquire(timeout=self._timeout):
            raise LockTimeout()
        try:
            if self._depth == 0:
                flags = (os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
                         | getattr(os, "O_BINARY", 0))
                fd = os.open(self.path, flags, 0o600)
                try:
                    delay = 0.005
                    while not _try_lock(fd):
                        if time.monotonic() >= deadline:
                            raise LockTimeout()
                        time.sleep(delay)
                        delay = min(delay * 2, 0.05)
                except BaseException:
                    os.close(fd)
                    raise
                self._fd = fd
            self._depth += 1
        except BaseException:
            self._thread_lock.release()
            raise

    def release(self) -> None:
        self._depth -= 1
        if self._depth == 0:
            fd, self._fd = self._fd, None
            if fd is not None:
                _unlock(fd)
                os.close(fd)
        self._thread_lock.release()

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, *exc_info) -> None:
        self.release()


# --- state ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class Actor:
    """Who asks: the client address, the Host it used, the credential's display name and which
    credential it is (a page key's id, DEPLOYMENT_KEY_ID, or "" for none or a one-time code)."""

    ip: str = ""
    host: str = ""
    credential: str = ""
    credential_id: str = ""

    def as_dict(self) -> dict:
        record = {"ip": _clip(self.ip, 64), "host": _clip(self.host, 200),
                  "credential": _clip(self.credential, 80)}
        if self.credential_id:
            record["credential_id"] = _clip(self.credential_id, 40)
        return record


@dataclass(frozen=True)
class MountStatus:
    state: str
    writable: bool
    directory: str
    detail: str


@dataclass(frozen=True)
class Pending:
    """A commit-confirm record: undone at `deadline` unless confirmed."""

    revision: int
    deadline: float
    previous: Mapping[str, Any]

    def as_dict(self) -> dict:
        return {"revision": self.revision, "deadline": self.deadline,
                "previous": {key: _jsonable(value) for key, value in self.previous.items()}}


@dataclass(frozen=True)
class PreferencesState:
    revision: int
    values: Mapping[str, Any]
    pending: Optional[Pending]
    saved_at: Optional[str]
    saved_by: Optional[Mapping[str, str]]
    source: str
    damaged: bool
    problems: tuple[str, ...]
    dropped: Mapping[str, str]


@dataclass(frozen=True)
class KeyRecord:
    id: str
    name: str
    role: str
    sha256: str
    hint: str
    created_at: str

    def public(self) -> dict:
        """What may leave the server: everything but the digest."""
        return {"id": self.id, "name": self.name, "role": self.role, "hint": self.hint,
                "created_at": self.created_at}

    def as_document(self) -> dict:
        return {**self.public(), "sha256": self.sha256}


@dataclass(frozen=True)
class KeysState:
    revision: int
    require_key: bool
    deployment_key_role: str
    keys: tuple[KeyRecord, ...]
    fail_closed: bool = False
    problem: Optional[str] = None

    def find(self, presented: str) -> Optional[KeyRecord]:
        """The page key `presented` is, compared in constant time against every digest."""
        if not isinstance(presented, str) or not presented:
            return None
        digest = key_digest(presented)
        found = None
        for record in self.keys:
            if hmac.compare_digest(digest, record.sha256):
                found = record
        return found

    def has_admin(self, deployment_key_present: bool) -> bool:
        """Is the server claimed: an admin page key, or the YAML key in role admin?"""
        if any(record.role == schema.ROLE_ADMIN for record in self.keys):
            return True
        return (bool(deployment_key_present) and not self.fail_closed
                and self.deployment_key_role == schema.ROLE_ADMIN)

    def effective_require_key(self, deployment_key_present: bool) -> bool:
        return self.fail_closed or self.require_key or bool(deployment_key_present)

    def public(self) -> dict:
        return {"revision": self.revision, "require_key": self.require_key,
                "deployment_key_role": self.deployment_key_role,
                "keys": [record.public() for record in self.keys],
                "fail_closed": self.fail_closed, "problem": self.problem}


@dataclass(frozen=True)
class SettingsState:
    mount: MountStatus
    safe_mode: bool
    preferences: PreferencesState
    keys: KeysState


@dataclass(frozen=True)
class HistoryEntry:
    id: str
    revision: int
    saved_at: Optional[str]
    saved_by: Optional[Mapping[str, str]]
    event: str
    changed: tuple[str, ...]

    def as_dict(self) -> dict:
        return {"id": self.id, "revision": self.revision, "saved_at": self.saved_at,
                "saved_by": dict(self.saved_by) if self.saved_by else None, "event": self.event,
                "changed": list(self.changed)}


@dataclass(frozen=True)
class WriteResult:
    revision: int
    changed: tuple[str, ...]
    values: Mapping[str, Any]
    pending: Optional[Pending]
    dropped: Mapping[str, str]
    dry_run: bool
    written: bool


_EMPTY: Mapping[str, Any] = MappingProxyType({})
_ABSENT_PREFERENCES = PreferencesState(
    revision=0, values=_EMPTY, pending=None, saved_at=None, saved_by=None, source=SOURCE_ABSENT,
    damaged=False, problems=(), dropped=_EMPTY)
_DEFAULT_KEYS = KeysState(revision=0, require_key=False, deployment_key_role=schema.ROLE_ADMIN, keys=())

_HISTORY_NAME = re.compile(r"(\d{8}T\d{6}Z)-r(\d{1,15})\.json")
_HISTORY_ID = re.compile(r"\d{8}T\d{6}Z-r\d{1,15}")
_KEY_ID = re.compile(r"[A-Za-z0-9_-]{1,32}")
_HEX64 = re.compile(r"[0-9a-f]{64}")


# --- parsing the files ----------------------------------------------------------------------

def _clean_values(raw: Mapping) -> tuple[dict, dict]:
    """(valid values in schema order, {key: reason} for dropped ones); unknown keys are ignored."""
    values: dict[str, Any] = {}
    dropped: dict[str, str] = {}
    for key in schema.PREFERENCE_KEYS:
        if key not in raw:
            continue
        try:
            values[key] = schema.parse_value(key, raw[key])
        except schema.ValueInvalid as exc:
            dropped[key] = exc.message
    return values, dropped


def _clean_saved_by(raw: Any) -> Optional[Mapping[str, str]]:
    if not isinstance(raw, dict):
        return None
    return MappingProxyType({field: _clip(raw.get(field), 200)
                             for field in ("ip", "host", "credential", "credential_id")
                             if isinstance(raw.get(field), str)})


def _parse_pending(raw: Any) -> Optional[Pending]:
    if not isinstance(raw, dict):
        raise _Damaged("not an object")
    revision, deadline, previous = raw.get("revision"), raw.get("deadline"), raw.get("previous")
    if not _is_int(revision) or revision < 0 or not _is_number(deadline) or not isinstance(previous, dict):
        raise _Damaged("incomplete")
    clean: dict[str, Any] = {}
    for key in sorted(schema.COMMIT_CONFIRM_KEYS):
        if key not in previous:
            continue
        value = previous[key]
        if value is None:
            clean[key] = None
            continue
        try:
            clean[key] = schema.parse_value(key, value)
        except schema.ValueInvalid:
            # Unusable today: the lenient reader would have dropped it, i.e. followed the YAML.
            clean[key] = None
    if not clean:
        return None
    return Pending(revision=revision, deadline=float(deadline), previous=MappingProxyType(clean))


def _preferences_from_document(document: Any, *, source: str) -> PreferencesState:
    if not isinstance(document, dict):
        raise _Damaged("not a JSON object")
    version, revision = document.get("schema"), document.get("revision")
    if not _is_int(version) or version < 1:
        raise _Damaged("no schema version")
    if not _is_int(revision) or revision < 0:
        raise _Damaged("no revision")
    raw_values = document.get("values", {})
    if not isinstance(raw_values, dict):
        raise _Damaged("values is not an object")
    values, dropped = _clean_values(raw_values)
    problems = [f"{key} in gateway.json is ignored: {reason}" for key, reason in dropped.items()]
    pending = None
    if document.get("pending") is not None:
        try:
            pending = _parse_pending(document["pending"])
        except _Damaged as exc:
            problems.append(f"The pending confirmation in gateway.json is ignored ({exc}).")
    saved_at = document.get("saved_at")
    return PreferencesState(
        revision=revision, values=MappingProxyType(values), pending=pending,
        saved_at=_clip(saved_at, 40) if isinstance(saved_at, str) else None,
        saved_by=_clean_saved_by(document.get("saved_by")), source=source, damaged=False,
        problems=tuple(problems), dropped=MappingProxyType(dropped))


def _key_record(raw: Any) -> KeyRecord:
    if not isinstance(raw, dict):
        raise _Damaged("a key entry is not an object")
    key_id, role, digest = raw.get("id"), raw.get("role"), raw.get("sha256")
    hint, created_at = raw.get("hint", ""), raw.get("created_at", "")
    if not isinstance(key_id, str) or not _KEY_ID.fullmatch(key_id):
        raise _Damaged("a key entry has no valid id")
    try:
        name = schema.validate_key_name(raw.get("name"))
        schema.validate_role(role)
    except schema.ValueInvalid:
        raise _Damaged("a key entry has no valid name or role") from None
    if not isinstance(digest, str) or not _HEX64.fullmatch(digest):
        raise _Damaged("a key entry has no valid digest")
    if not isinstance(hint, str) or len(hint) > 16 or not isinstance(created_at, str) or len(created_at) > 40:
        raise _Damaged("a key entry is malformed")
    return KeyRecord(id=key_id, name=name, role=role, sha256=digest, hint=hint, created_at=created_at)


def _keys_from_document(document: Any) -> KeysState:
    if not isinstance(document, dict):
        raise _Damaged("not a JSON object")
    if document.get("schema") != FILE_SCHEMA or not _is_int(document.get("schema")):
        # A newer layout may restrict access in ways this version would not enforce.
        raise _Damaged("unknown schema version")
    revision, require_key = document.get("revision"), document.get("require_key")
    role, raw_keys = document.get("deployment_key_role"), document.get("keys")
    if not _is_int(revision) or revision < 0:
        raise _Damaged("no revision")
    if not isinstance(require_key, bool):
        raise _Damaged("require_key is not true or false")
    if role not in schema.ROLES:
        raise _Damaged("deployment_key_role is not admin or client")
    if not isinstance(raw_keys, list) or len(raw_keys) > 1024:
        raise _Damaged("keys is not a list")
    records = tuple(_key_record(raw) for raw in raw_keys)
    if len({r.id for r in records}) != len(records) or len({r.sha256 for r in records}) != len(records):
        raise _Damaged("a key appears twice")
    return KeysState(revision=revision, require_key=require_key, deployment_key_role=role, keys=records)


def _changed(old: Mapping[str, Any], new: Mapping[str, Any]) -> tuple[str, ...]:
    out = []
    for key in schema.PREFERENCE_KEYS:
        if (key in old) != (key in new):
            out.append(key)
        elif key in old and not schema.same_value(old[key], new[key]):
            out.append(key)
    return tuple(out)


def _changes(old: Mapping[str, Any], new: Mapping[str, Any], keys: Iterable[str]) -> dict:
    return {key: [_jsonable(old.get(key)), _jsonable(new.get(key))] for key in keys}


def _ordered(values: Mapping[str, Any]) -> dict:
    return {key: values[key] for key in schema.PREFERENCE_KEYS if key in values}


def _pending_after(current: PreferencesState, new_values: Mapping[str, Any], changed: Iterable[str],
                   revision: int, deadline: float) -> Optional[Pending]:
    """The commit-confirm record after a write: created, extended or carried over unchanged."""
    confirm_keys = [key for key in changed if key in schema.COMMIT_CONFIRM_KEYS]
    existing = current.pending
    if not confirm_keys:
        return existing
    previous = dict(existing.previous) if existing else {}
    for key in confirm_keys:
        # The oldest value wins: an undo goes back to what was confirmed last.
        previous.setdefault(key, current.values.get(key))
    previous = {key: value for key, value in previous.items()
                if not (value is None and key not in new_values)
                and not (value is not None and key in new_values and schema.same_value(value, new_values[key]))}
    if not previous:
        return None
    return Pending(revision=revision, deadline=deadline, previous=MappingProxyType(dict(sorted(previous.items()))))


def _canonical_code(code: Any) -> Optional[str]:
    if not isinstance(code, str) or len(code) > 64:
        return None
    text = re.sub(r"[\s-]", "", code).upper().translate(str.maketrans({"O": "0", "I": "1", "L": "1"}))
    if len(text) != CODE_LENGTH or any(ch not in CODE_ALPHABET for ch in text):
        return None
    return text


def _code_digest(canonical: str) -> str:
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _format_code(canonical: str) -> str:
    return "-".join(canonical[i:i + 5] for i in range(0, CODE_LENGTH, 5))


def _write_operation(method: Callable) -> Callable:
    """I/O failures of a write become WriteFailed; the log names only the operation."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except OSError as exc:
            logger.error("settings: %s failed (%s)", method.__name__.lstrip("_"), type(exc).__name__)
            raise WriteFailed() from exc

    return wrapper


# --- the store ------------------------------------------------------------------------------

class SettingsStore:
    """See the module docstring. One instance per process; all methods are synchronous."""

    def __init__(self, directory: Any = DEFAULT_DIRECTORY, *, verify_mount: bool = True,
                 mountinfo_path: str = MOUNTINFO_PATH, lock_path: Optional[str] = None,
                 clock: Callable[[], float] = time.time, confirm_window: float = CONFIRM_WINDOW_S,
                 lock_timeout: float = LOCK_TIMEOUT_S):
        self.directory = Path(directory)
        self.verify_mount = verify_mount
        self._mountinfo_path = mountinfo_path
        self._lock = FileLock(lock_path, timeout=lock_timeout)
        self._clock = clock
        self._confirm_window = confirm_window
        self._mutex = threading.RLock()
        self._mount_checked = False
        self._mount_entry: Optional[MountEntry] = None
        self._state: Optional[SettingsState] = None
        # The stat signature of each file as last read with a definite result. A read that failed
        # for a reason that may pass leaves it alone, so the file is read again (after READ_RETRY_S).
        self._signatures: dict[str, Any] = {}
        self._read_failures: dict[str, str] = {}      # file -> why its last read failed and may pass
        self._read_retry_at: dict[str, float] = {}    # file -> time.monotonic() of the next attempt
        self._last_good_preferences: Optional[PreferencesState] = None
        self._keys_revision_floor = 0
        self.reads = 0

    @classmethod
    def from_environment(cls, environ: Optional[Mapping[str, str]] = None, **kwargs: Any) -> "SettingsStore":
        """/app/settings with the mount check, or the TTS_STT_SETTINGS_DIR folder without it."""
        if environ is None:
            override = os.environ.get("TTS_STT_SETTINGS_DIR", "")
        else:
            override = environ.get(OVERRIDE_ENV, "")
        override = (override or "").strip()
        if override:
            return cls(override, verify_mount=False, **kwargs)
        return cls(DEFAULT_DIRECTORY, verify_mount=True, **kwargs)

    # --- paths and low-level I/O ---------------------------------------------------------

    def _path(self, name: str) -> Path:
        return self.directory / name

    def _history_path(self, entry_id: str) -> Path:
        return self.directory / HISTORY_DIR / f"{entry_id}.json"

    @staticmethod
    def _signature(path: Path) -> Any:
        """What tells a changed file: os.replace() gives a new inode, and the owner, mode and
        ctime show a repair by chmod/chown/setfacl of a file that could not be read."""
        try:
            st = os.stat(path, follow_symlinks=False)
        except (FileNotFoundError, NotADirectoryError):
            return None
        except OSError as exc:
            return ("error", exc.errno)
        return (st.st_ino, st.st_mtime_ns, st.st_size, st.st_mode, st.st_uid, st.st_gid, st.st_ctime_ns)

    def _read_bytes(self, path: Path) -> bytes:
        """A regular file's bytes, never through a symbolic link; raises _Damaged or OSError."""
        if not hasattr(os, "O_NOFOLLOW") and os.path.islink(path):
            raise _Damaged("is a symbolic link")
        flags = (os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
                 | getattr(os, "O_BINARY", 0))
        try:
            fd = os.open(path, flags)
        except OSError as exc:
            if exc.errno in (errno.ELOOP, errno.EMLINK):
                raise _Damaged("is a symbolic link") from None
            raise
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise _Damaged("is not a regular file")
            if info.st_size > MAX_FILE_BYTES:
                raise _Damaged("is too large")
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_FILE_BYTES:
                    raise _Damaged("is too large")
                chunks.append(chunk)
        finally:
            os.close(fd)
            self.reads += 1
        return b"".join(chunks)

    def _atomic_write(self, path: Path, payload: Any) -> Any:
        """Temp file, fsync, os.replace, folder fsync; returns the new file's signature."""
        data = payload if isinstance(payload, bytes) else _dump(payload)
        directory = os.fspath(path.parent)
        fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=directory)
        try:
            try:
                view = memoryview(data)
                while view:
                    written = os.write(fd, view)
                    view = view[written:]
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(temp, path)
        except BaseException:
            try:
                os.unlink(temp)
            except OSError:
                pass
            raise
        _fsync_directory(directory)
        return self._signature(path)

    def _keep_damaged_copy(self, name: str) -> None:
        """Before a write replaces a damaged file, keep it as <name>.damaged (best effort).

        A file this process may not read (its permissions) is kept as it is, under the new name,
        through a second hard link: the write that follows replaces only the old name.
        """
        source, target = self._path(name), self._path(f"{name}.damaged")
        try:
            data = self._read_bytes(source)
        except FileNotFoundError:
            return
        except (OSError, _Damaged):
            data = None
        try:
            if data is not None:
                self._atomic_write(target, data)
                return
            temp = self._path(f".{name}.damaged.{secrets.token_hex(4)}.tmp")
            os.link(source, temp, follow_symlinks=False)
            try:
                os.replace(temp, target)
            except OSError:
                os.unlink(temp)
                raise
        except (OSError, NotImplementedError):
            pass

    # --- the mount ---------------------------------------------------------------------------

    def _mounted_entry(self) -> Optional[MountEntry]:
        if not self._mount_checked:
            entry = None
            try:
                with open(self._mountinfo_path, encoding="utf-8", errors="surrogateescape") as handle:
                    entry = find_mount(parse_mountinfo(handle.read()), os.fspath(self.directory))
            except OSError:
                entry = None
            self._mount_entry, self._mount_checked = entry, True
        return self._mount_entry

    def mount_status(self) -> MountStatus:
        """Whether writes are allowed here, and why (mountinfo is read once, the folder each time)."""
        directory = os.fspath(self.directory)
        exists = os.path.isdir(directory)
        if not self.verify_mount:
            if exists:
                return MountStatus(MOUNT_OVERRIDE, True, directory,
                                   "TTS_STT_SETTINGS_DIR names this folder; the mount check is skipped.")
            return MountStatus(MOUNT_MISSING, False, directory, "The settings folder does not exist.")
        entry = self._mounted_entry()
        if entry is None:
            return MountStatus(MOUNT_NOT_MOUNTED, False, directory,
                               "The settings folder is not mounted from the app's dataset.")
        if entry.read_only:
            return MountStatus(MOUNT_READ_ONLY, False, directory, "The settings folder is mounted read-only.")
        if not exists:
            return MountStatus(MOUNT_MISSING, False, directory, "The settings folder does not exist.")
        return MountStatus(MOUNT_MOUNTED, True, directory, "The settings folder is mounted from the app's dataset.")

    # --- reading -------------------------------------------------------------------------------

    @property
    def state(self) -> SettingsState:
        with self._mutex:
            if self._state is None:
                self.refresh()
            return self._state

    def refresh(self, force: bool = False) -> bool:
        """Stat the files; re-read the ones that changed. True when the state object changed.

        `force` re-reads gateway.json and keys.json whatever was cached (the write section does,
        under the lock); a state equal to the one in force is kept as the same object.
        """
        with self._mutex:
            old = self._state
            mount = self.mount_status()
            if old is not None and mount == old.mount:
                mount = old.mount
            safe_mode = os.path.lexists(self._path(SAFE_MODE_FILE))
            preferences = self._refresh_preferences(old.preferences if old else None, force)
            keys = self._refresh_keys(old.keys if old else None, force)
            if (old is not None and mount is old.mount and safe_mode == old.safe_mode
                    and preferences is old.preferences and keys is old.keys):
                return False
            self._state = SettingsState(mount=mount, safe_mode=safe_mode, preferences=preferences, keys=keys)
            return True

    def _unchanged(self, name: str, signature: Any, previous: Any, force: bool) -> bool:
        """Can `previous` stand without a read: the same file, or a failed read not yet due again?"""
        if previous is None or force:
            return False
        if signature == self._signatures.get(name):
            return True
        retry_at = self._read_retry_at.get(name)
        return retry_at is not None and time.monotonic() < retry_at

    def _read_settled(self, name: str, signature: Any) -> None:
        """The read had a definite result (the content, or a verdict on it): remember the file."""
        self._signatures[name] = signature
        self._read_failures.pop(name, None)
        self._read_retry_at.pop(name, None)

    def _read_failed(self, name: str, exc: OSError) -> None:
        """The read failed for a reason that may pass: remember nothing, and try again later."""
        reason = _reason(exc)
        if self._read_failures.get(name) != reason:
            logger.warning("settings: %s could not be read (%s); it is read again on a later request", name, reason)
        self._read_failures[name] = reason
        self._read_retry_at[name] = time.monotonic() + READ_RETRY_S

    def _refresh_preferences(self, previous: Optional[PreferencesState], force: bool = False) -> PreferencesState:
        path = self._path(PREFERENCES_FILE)
        signature = self._signature(path)
        if self._unchanged(PREFERENCES_FILE, signature, previous, force):
            return previous
        try:
            if signature is None:
                raise FileNotFoundError(errno.ENOENT, PREFERENCES_FILE)
            state = _preferences_from_document(schema.parse_json_strict(self._read_bytes(path)),
                                               source=SOURCE_FILE)
        except FileNotFoundError:
            state, signature = _ABSENT_PREFERENCES, None
            self._last_good_preferences = state
        except OSError as exc:
            if _may_pass(exc):
                self._read_failed(PREFERENCES_FILE, exc)
                # Whatever was in force stays until the file can be read (a process that has
                # nothing yet starts from its fallbacks, as for a damaged file).
                if previous is not None:
                    return previous
                return self._damaged_preferences(_reason(exc), unread=True)
            state = self._damaged_preferences(_reason(exc))
        except (ValueError, _Damaged) as exc:
            state = self._damaged_preferences(_reason(exc))
        else:
            if state.problems:
                logger.warning("settings: %s", "; ".join(state.problems))
            self._last_good_preferences = state
        self._read_settled(PREFERENCES_FILE, signature)
        return previous if previous is not None and state == previous else state

    def _damaged_preferences(self, reason: str, *, unread: bool = False) -> PreferencesState:
        problem = f"gateway.json could not be read ({reason})" if unread else f"gateway.json is damaged ({reason})"
        log = logger.warning if unread else logger.error
        good = self._last_good_preferences
        if good is not None:
            log("settings: %s; keeping what was in force before", problem)
            if good.source == SOURCE_ABSENT:
                kept = "the app YAML values stay in force."
            else:
                kept = f"the values saved as revision {good.revision} stay in force."
            return replace(good, source=SOURCE_LAST_GOOD, damaged=True,
                           problems=(f"{problem}; {kept}",) + good.problems)
        fallback = self._newest_history_state()
        if fallback is not None:
            log("settings: %s; using history revision %d", problem, fallback.revision)
            return replace(fallback, source=SOURCE_HISTORY, damaged=True,
                           problems=(f"{problem}; the saved version {fallback.revision} from the history "
                                     "is in force.",) + fallback.problems)
        log("settings: %s; using the app YAML values", problem)
        return replace(_ABSENT_PREFERENCES, source=SOURCE_NONE, damaged=True,
                       problems=(f"{problem}; the app YAML values are in force.",))

    def _newest_history_state(self) -> Optional[PreferencesState]:
        for _revision, _stamp_text, entry_id in self._history_entries():
            try:
                document = schema.parse_json_strict(self._read_bytes(self._history_path(entry_id)))
                state = _preferences_from_document(document, source=SOURCE_HISTORY)
            except (OSError, ValueError, _Damaged):
                continue
            return replace(state, pending=None)
        return None

    def _refresh_keys(self, previous: Optional[KeysState], force: bool = False) -> KeysState:
        path = self._path(KEYS_FILE)
        signature = self._signature(path)
        if self._unchanged(KEYS_FILE, signature, previous, force):
            return previous
        try:
            if signature is None:
                raise FileNotFoundError(errno.ENOENT, KEYS_FILE)
            state = _keys_from_document(schema.parse_json_strict(self._read_bytes(path)))
        except FileNotFoundError:
            state, signature = _DEFAULT_KEYS, None
        except OSError as exc:
            if _may_pass(exc):
                # The change that could not be read may be a revocation: fail closed meanwhile.
                self._read_failed(KEYS_FILE, exc)
                state = self._closed_keys(
                    f"keys.json could not be read ({_reason(exc)}): a key is required, and only the app YAML "
                    "key is accepted, until it can be read again.")
                return previous if previous is not None and state == previous else state
            state = self._closed_keys(self._damaged_keys_problem(_reason(exc)))
        except (ValueError, _Damaged) as exc:
            state = self._closed_keys(self._damaged_keys_problem(_reason(exc)))
        else:
            self._keys_revision_floor = max(self._keys_revision_floor, state.revision)
        self._read_settled(KEYS_FILE, signature)
        return previous if previous is not None and state == previous else state

    @staticmethod
    def _damaged_keys_problem(reason: str) -> str:
        logger.error("settings: keys.json is damaged (%s); a key is required and only the app "
                     "YAML key is accepted until it is repaired", reason)
        return (f"keys.json is damaged ({reason}): a key is required, and only the app YAML "
                "key is accepted, until a recovery code rewrites it or it is deleted.")

    def _closed_keys(self, problem: str) -> KeysState:
        """The fail-closed state: a key required, no page keys, the app YAML key a client."""
        return KeysState(revision=self._keys_revision_floor, require_key=True,
                         deployment_key_role=schema.ROLE_CLIENT, keys=(), fail_closed=True, problem=problem)

    def _history_entries(self) -> list[tuple[int, str, str]]:
        """(revision, stamp, id) of every history file, newest first."""
        directory = self._path(HISTORY_DIR)
        if os.path.islink(directory):
            return []
        try:
            names = os.listdir(directory)
        except OSError:
            return []
        found = []
        for name in names:
            match = _HISTORY_NAME.fullmatch(name)
            if match:
                found.append((int(match.group(2)), match.group(1), name[: -len(".json")]))
        found.sort(reverse=True)
        return found

    def history(self, limit: int = HISTORY_LIMIT) -> list[HistoryEntry]:
        """Summaries of the saved versions, newest first (unreadable ones are skipped)."""
        out: list[HistoryEntry] = []
        for revision, _stamp_text, entry_id in self._history_entries():
            if len(out) >= limit:
                break
            try:
                document = schema.parse_json_strict(self._read_bytes(self._history_path(entry_id)))
            except (OSError, ValueError, _Damaged):
                continue
            if not isinstance(document, dict):
                continue
            raw_changed = document.get("changed")
            changed = tuple(key for key in raw_changed if isinstance(key, str))[:64] if isinstance(raw_changed, list) else ()
            event, saved_at = document.get("event"), document.get("saved_at")
            out.append(HistoryEntry(
                id=entry_id, revision=revision,
                saved_at=_clip(saved_at, 40) if isinstance(saved_at, str) else None,
                saved_by=_clean_saved_by(document.get("saved_by")),
                event=_clip(event, 40) if isinstance(event, str) else "save", changed=changed))
        return out

    # --- writing: common ------------------------------------------------------------------------

    @contextmanager
    def _exclusive(self, authorize: Optional[Callable[[KeysState], None]] = None):
        """The write section: a writable folder, this process's mutex, the cross-process lock,
        and a state that is fresh under that lock.

        Fresh means read from the files now, not taken from this process's cache: a worker that
        missed a change (a read that failed) must not decide from what it had before. A read that
        fails again refuses the write (Unreadable). `authorize(keys)` then judges the credential
        the write was let in with against the keys just read, so a key revoked or made a client
        by any worker since the request arrived writes nothing.
        """
        status = self.mount_status()
        if not status.writable:
            raise NotWritable()
        with self._mutex:
            self._lock.acquire()
            try:
                self.refresh(force=True)
                if self._read_failures:
                    raise Unreadable()
                if authorize is not None:
                    authorize(self._state.keys)
                yield
            finally:
                self._lock.release()

    def _commit_preferences(self, state: PreferencesState, signature: Any) -> None:
        self._signatures[PREFERENCES_FILE] = signature
        self._last_good_preferences = state
        self._state = replace(self._state, preferences=state)

    def _commit_keys(self, state: KeysState, signature: Any) -> None:
        self._signatures[KEYS_FILE] = signature
        self._keys_revision_floor = max(self._keys_revision_floor, state.revision)
        self._state = replace(self._state, keys=state)

    def _max_history_revision(self) -> int:
        entries = self._history_entries()
        return entries[0][0] if entries else 0

    def _write_history(self, document: Mapping, event: str, changed: Iterable[str], now: float) -> None:
        directory = self._path(HISTORY_DIR)
        try:
            if os.path.islink(directory):
                raise OSError(errno.ELOOP, "history is a symbolic link")
            if not os.path.isdir(directory):
                os.mkdir(directory, 0o700)
                os.chmod(directory, 0o700)
            entry = {
                "schema": FILE_SCHEMA, "revision": document["revision"],
                "saved_at": document["saved_at"], "saved_by": document["saved_by"], "event": event,
                "changed": list(changed), "values": document["values"],
            }
            self._atomic_write(directory / f"{_stamp(now)}-r{document['revision']}.json", entry)
            for _revision, _stamp_text, entry_id in self._history_entries()[HISTORY_LIMIT:]:
                try:
                    os.unlink(self._history_path(entry_id))
                except OSError:
                    pass
        except OSError as exc:
            logger.warning("settings: could not write the history (%s)", type(exc).__name__)

    def _audit_line(self, event: str, actor: Optional[Actor], result: str, fields: Mapping[str, Any]) -> str:
        record: dict[str, Any] = {"time": _iso(self._clock()), "worker": os.getpid(),
                                  "event": _clip(event, 40), "result": _clip(result, 80)}
        if actor is not None:
            record.update(actor.as_dict())
        for name, value in fields.items():
            if name not in record:
                record[name] = value
        return json.dumps(_scrub(record), ensure_ascii=True, separators=(",", ":"), default=str)

    def _append_audit(self, line: str) -> None:
        path = self._path(AUDIT_FILE)
        data = (line + "\n").encode("utf-8")
        try:
            size = os.stat(path, follow_symlinks=False).st_size
        except FileNotFoundError:
            size = 0
        if size and size + len(data) > AUDIT_MAX_BYTES:
            for index in range(AUDIT_BACKUPS, 0, -1):
                source = path if index == 1 else self._path(f"{AUDIT_FILE}.{index - 1}")
                if os.path.lexists(source):
                    os.replace(source, self._path(f"{AUDIT_FILE}.{index}"))
        flags = (os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
                 | getattr(os, "O_BINARY", 0))
        fd = os.open(path, flags, 0o600)
        try:
            view = memoryview(data)
            while view:
                written = os.write(fd, view)
                view = view[written:]
        finally:
            os.close(fd)

    def _audit_locked(self, event: str, actor: Optional[Actor], result: str = "ok", **fields: Any) -> None:
        """An audit line from inside a write section (the lock is held)."""
        line = self._audit_line(event, actor, result, fields)
        audit_logger.info("%s", line)
        try:
            self._append_audit(line)
        except OSError as exc:
            logger.warning("settings: could not append to the audit log (%s)", type(exc).__name__)

    def audit(self, event: str, *, actor: Optional[Actor] = None, result: str = "ok", to_file: bool = True,
              **fields: Any) -> None:
        """One audit line: to the logger always, to audit.jsonl when the folder is writable.

        `to_file=False` is for what anyone can cause as often as they like (a refusal that needs no
        admin key): the line goes to the logger only, takes no lock and touches no file, so a flood
        of them cannot rotate the saves, claims and key changes out of audit.jsonl.

        Fields named like a secret (key, code, sha256, token, ...) are left out, and anything
        shaped like a page key, a code or a digest is masked. Never raises.
        """
        try:
            line = self._audit_line(event, actor, result, fields)
        except Exception:  # pragma: no cover - only an exotic field type gets here
            line = json.dumps({"event": _clip(event, 40), "result": _clip(result, 80)})
        audit_logger.info("%s", line)
        if not to_file:
            return
        try:
            if not self.mount_status().writable:
                return
            with self._mutex:
                self._lock.acquire()
                try:
                    self._append_audit(line)
                finally:
                    self._lock.release()
        except Exception as exc:
            logger.warning("settings: could not append to the audit log (%s)", type(exc).__name__)

    # --- writing: preferences -----------------------------------------------------------------

    def _preferences_document(self, revision: int, saved_at: Optional[str], saved_by: Mapping,
                              values: Mapping[str, Any], pending: Optional[Pending]) -> dict:
        return {
            "schema": FILE_SCHEMA, "revision": revision, "saved_at": saved_at,
            "saved_by": dict(saved_by), "values": {k: _jsonable(v) for k, v in _ordered(values).items()},
            "pending": pending.as_dict() if pending else None,
        }

    def _store_preferences(self, document: dict, pending: Optional[Pending], values: Mapping[str, Any],
                           *, keep_damaged: bool) -> PreferencesState:
        if keep_damaged:
            self._keep_damaged_copy(PREFERENCES_FILE)
        signature = self._atomic_write(self._path(PREFERENCES_FILE), document)
        state = PreferencesState(
            revision=document["revision"], values=MappingProxyType(_ordered(values)), pending=pending,
            saved_at=document["saved_at"], saved_by=MappingProxyType(dict(document["saved_by"])),
            source=SOURCE_FILE, damaged=False, problems=(), dropped=_EMPTY)
        self._commit_preferences(state, signature)
        return state

    def _expire_locked(self) -> bool:
        current = self._state.preferences
        pending = current.pending
        now = self._clock()
        if pending is None or now < pending.deadline:
            return False
        values = dict(current.values)
        for key, previous in pending.previous.items():
            if previous is None:
                values.pop(key, None)
            else:
                values[key] = previous
        changed = _changed(current.values, values)
        revision = max(current.revision, self._max_history_revision()) + 1
        actor = Actor(credential=AUTO_REVERT_ACTOR)
        document = self._preferences_document(revision, _iso(now), actor.as_dict(), values, None)
        self._store_preferences(document, None, values, keep_damaged=current.damaged)
        self._write_history(document, "auto-revert", changed, now)
        self._audit_locked("auto-reverted", None, revision=revision, reverted_revision=pending.revision,
                           changes=_changes(current.values, values, changed))
        return True

    @_write_operation
    def _write_preferences(self, build: Callable[[PreferencesState], tuple[dict, dict]], *, base_revision: int,
                           actor: Actor, check: Optional[Callable], dry_run: bool, event: str,
                           audit_fields: Optional[Mapping[str, Any]] = None,
                           authorize: Optional[Callable[[KeysState], None]] = None) -> WriteResult:
        if not _is_int(base_revision):
            raise InvalidInput({"base_revision": "Expected the revision the change is based on."})
        with self._exclusive(authorize):
            if self._state.preferences.pending is not None:
                # An overdue confirmation is undone first; the caller's revision is then stale.
                self._expire_locked()
            current = self._state.preferences
            if base_revision != current.revision:
                raise RevisionConflict(current.revision)
            values, dropped = build(current)
            values = _ordered(values)
            frozen = MappingProxyType(values)
            changed = _changed(current.values, values)
            if not changed:
                return WriteResult(revision=current.revision, changed=(), values=current.values,
                                   pending=current.pending, dropped=MappingProxyType(dropped),
                                   dry_run=dry_run, written=False)
            if check is not None:
                check(frozen, changed)
            now = self._clock()
            revision = max(current.revision, self._max_history_revision()) + 1
            pending = _pending_after(current, values, changed, revision, now + self._confirm_window)
            if dry_run:
                return WriteResult(revision=revision, changed=changed, values=frozen, pending=pending,
                                   dropped=MappingProxyType(dropped), dry_run=True, written=False)
            document = self._preferences_document(revision, _iso(now), actor.as_dict(), values, pending)
            self._store_preferences(document, pending, values, keep_damaged=current.damaged)
            self._write_history(document, event, changed, now)
            self._audit_locked(event, actor, revision=revision,
                               changes=_changes(current.values, values, changed), **(audit_fields or {}))
            return WriteResult(revision=revision, changed=changed, values=frozen, pending=pending,
                               dropped=MappingProxyType(dropped), dry_run=False, written=True)

    def update_preferences(self, set_values: Optional[Mapping[str, Any]] = None, reset: Iterable[str] = (), *,
                           base_revision: int, actor: Actor, check: Optional[Callable] = None,
                           dry_run: bool = False,
                           authorize: Optional[Callable[[KeysState], None]] = None) -> WriteResult:
        """Set some overrides and remove others ("Reset to YAML"); every value is re-checked."""
        if set_values is not None and not isinstance(set_values, Mapping):
            raise InvalidInput({"set": "Expected an object of settings."})
        if isinstance(reset, (str, bytes)) or not isinstance(reset, Iterable):
            raise InvalidInput({"reset": "Expected a list of settings."})
        set_values = dict(set_values or {})
        reset = tuple(reset)
        errors: dict[str, str] = {}
        clean: dict[str, Any] = {}

        def unknown(key: Any) -> None:
            known = isinstance(key, str) and key in schema.BY_KEY
            errors[str(key)[:64]] = "This setting is not stored here." if known else "Unknown setting."

        for key, raw in set_values.items():
            if key not in schema.PREFERENCE_KEYS:
                unknown(key)
                continue
            try:
                clean[key] = schema.parse_value(key, raw)
            except schema.ValueInvalid as exc:
                errors[key] = exc.message
        for key in reset:
            if key not in schema.PREFERENCE_KEYS:
                unknown(key)
            elif key in set_values:
                errors[key] = "Set and reset at the same time."
        if errors:
            raise InvalidInput(errors)

        def build(current: PreferencesState) -> tuple[dict, dict]:
            values = {key: value for key, value in current.values.items() if key not in reset}
            values.update(clean)
            return values, {}

        return self._write_preferences(build, base_revision=base_revision, actor=actor, check=check,
                                       dry_run=dry_run, event="save", authorize=authorize)

    def restore(self, history_id: str, *, base_revision: int, actor: Actor, check: Optional[Callable] = None,
                dry_run: bool = False, authorize: Optional[Callable[[KeysState], None]] = None) -> WriteResult:
        """Make a saved version current again, as a new revision (values invalid today are dropped)."""
        if not isinstance(history_id, str) or not _HISTORY_ID.fullmatch(history_id):
            raise HistoryNotFound()

        def build(_current: PreferencesState) -> tuple[dict, dict]:
            try:
                document = schema.parse_json_strict(self._read_bytes(self._history_path(history_id)))
                state = _preferences_from_document(document, source=SOURCE_HISTORY)
            except (OSError, ValueError, _Damaged):
                raise HistoryNotFound() from None
            return dict(state.values), dict(state.dropped)

        return self._write_preferences(build, base_revision=base_revision, actor=actor, check=check,
                                       dry_run=dry_run, event="restore", audit_fields={"history_id": history_id},
                                       authorize=authorize)

    def discard(self, *, base_revision: int, actor: Actor, check: Optional[Callable] = None,
                dry_run: bool = False, authorize: Optional[Callable[[KeysState], None]] = None) -> WriteResult:
        """Drop every override (the app YAML values apply); the discarded version stays in the history."""
        return self._write_preferences(lambda _current: ({}, {}), base_revision=base_revision, actor=actor,
                                       check=check, dry_run=dry_run, event="discard", authorize=authorize)

    @_write_operation
    def confirm(self, revision: int, *, actor: Optional[Actor] = None,
                authorize: Optional[Callable[[KeysState], None]] = None) -> bool:
        """Clear the pending record created by `revision`; False when there is none or it is overdue."""
        with self._exclusive(authorize):
            current = self._state.preferences
            pending = current.pending
            if pending is None or not _is_int(revision) or revision != pending.revision:
                return False
            if self._clock() >= pending.deadline:
                self._expire_locked()
                return False
            document = self._preferences_document(current.revision, current.saved_at,
                                                  current.saved_by or {}, current.values, None)
            self._store_preferences(document, None, current.values, keep_damaged=current.damaged)
            self._audit_locked("confirmed", actor, revision=current.revision, settings=sorted(pending.previous))
            return True

    def expire_pending(self) -> bool:
        """Undo an unconfirmed commit-confirm change once its deadline has passed. Never raises."""
        try:
            state = self.state
            pending = state.preferences.pending
            if pending is None or self._clock() < pending.deadline or not state.mount.writable:
                return False
            with self._exclusive():
                return self._expire_locked()
        except Exception as exc:
            logger.error("settings: could not undo an unconfirmed change (%s)", type(exc).__name__)
            return False

    # --- writing: keys -----------------------------------------------------------------------

    def _usable_keys(self) -> KeysState:
        keys = self._state.keys
        if keys.fail_closed:
            raise KeysFileDamaged()
        return keys

    def _next_keys_revision(self, current: KeysState) -> int:
        return max(current.revision, self._keys_revision_floor) + 1

    def _store_keys(self, state: KeysState) -> None:
        document = {
            "schema": FILE_SCHEMA, "revision": state.revision, "require_key": state.require_key,
            "deployment_key_role": state.deployment_key_role,
            "keys": [record.as_document() for record in state.keys],
        }
        self._commit_keys(state, self._atomic_write(self._path(KEYS_FILE), document))

    @staticmethod
    def _new_key_id(existing: Iterable[KeyRecord]) -> str:
        taken = {record.id for record in existing}
        while True:
            candidate = secrets.token_hex(4)
            if candidate not in taken:
                return candidate

    @staticmethod
    def _check_new_key(existing: Iterable[KeyRecord], name: str, digest: str) -> None:
        errors = {}
        existing = tuple(existing)
        if any(record.name.casefold() == name.casefold() for record in existing):
            errors["name"] = "A key with this name exists."
        if any(hmac.compare_digest(record.sha256, digest) for record in existing):
            errors["key"] = "This key is already in use."
        if errors:
            raise InvalidInput(errors)

    @staticmethod
    def _checked_key_fields(name: Any, key: Any, role: Any = schema.ROLE_ADMIN) -> tuple[str, str]:
        """The new key's name and role, or InvalidInput. Checked before anything is locked or
        read, so a refused name costs no code.

        A name the audit trail uses for another credential ("deployment key", "one-time code",
        "auto-revert") is refused here, when a key is made, and never when keys.json is read: one
        such name in an older file must not fail the whole file closed.
        """
        errors = {}
        clean_name = clean_role = None
        try:
            clean_name = schema.validate_key_name(name)
            if _reserved_name(clean_name):
                errors["name"] = "This name is reserved for the audit trail; choose another."
        except schema.ValueInvalid as exc:
            errors["name"] = exc.message
        try:
            clean_role = schema.validate_role(role)
        except schema.ValueInvalid as exc:
            errors["role"] = exc.message
        try:
            schema.validate_page_key(key)
        except schema.ValueInvalid as exc:
            errors["key"] = exc.message
        if errors:
            raise InvalidInput(errors)
        return clean_name, clean_role

    @_write_operation
    def set_access(self, *, base_revision: int, actor: Actor, deployment_key_present: bool,
                   require_key: Optional[bool] = None, deployment_key_role: Optional[str] = None,
                   authorize: Optional[Callable[[KeysState], None]] = None) -> KeysState:
        """Open or key-required API, and the YAML key's role; refuses a change that leaves no admin."""
        errors = {}
        if require_key is not None and not isinstance(require_key, bool):
            errors["require_key"] = "Expected true or false."
        if deployment_key_role is not None:
            try:
                schema.validate_role(deployment_key_role)
            except schema.ValueInvalid as exc:
                errors["deployment_key_role"] = exc.message
        if not _is_int(base_revision):
            errors["base_revision"] = "Expected the revision the change is based on."
        if errors:
            raise InvalidInput(errors)
        with self._exclusive(authorize):
            current = self._usable_keys()
            if base_revision != current.revision:
                raise RevisionConflict(current.revision)
            candidate = replace(
                current,
                require_key=current.require_key if require_key is None else require_key,
                deployment_key_role=current.deployment_key_role if deployment_key_role is None else deployment_key_role)
            if (candidate.require_key == current.require_key
                    and candidate.deployment_key_role == current.deployment_key_role):
                return current
            if (candidate.deployment_key_role != current.deployment_key_role
                    and not candidate.has_admin(deployment_key_present)):
                raise LastAdminCredential(
                    "The key from the app YAML is the only admin credential; create an admin key first.")
            if candidate.require_key and not deployment_key_present and not candidate.keys:
                raise NoKeyConfigured()
            state = replace(candidate, revision=self._next_keys_revision(current))
            self._store_keys(state)
            self._audit_locked("access", actor, revision=state.revision, changes={
                "require_key": [current.require_key, state.require_key],
                "deployment_key_role": [current.deployment_key_role, state.deployment_key_role]})
            return state

    @_write_operation
    def add_key(self, *, name: str, role: str, key: str, actor: Actor,
                authorize: Optional[Callable[[KeysState], None]] = None) -> KeyRecord:
        """Store a page key as its digest and hint; the plaintext is not kept anywhere."""
        clean_name, clean_role = self._checked_key_fields(name, key, role)
        with self._exclusive(authorize):
            current = self._usable_keys()
            if len(current.keys) >= MAX_KEYS:
                raise KeyLimitReached()
            digest = key_digest(key)
            self._check_new_key(current.keys, clean_name, digest)
            record = KeyRecord(id=self._new_key_id(current.keys), name=clean_name, role=clean_role,
                               sha256=digest, hint=schema.key_hint(key), created_at=_iso(self._clock()))
            state = replace(current, revision=self._next_keys_revision(current), keys=current.keys + (record,))
            self._store_keys(state)
            self._audit_locked("key-created", actor, key_id=record.id, name=record.name, role=record.role,
                               hint=record.hint)
            return record

    @_write_operation
    def revoke_key(self, key_id: str, *, actor: Actor, deployment_key_present: bool,
                   authorize: Optional[Callable[[KeysState], None]] = None) -> KeyRecord:
        """Remove a page key; the last admin credential cannot go."""
        with self._exclusive(authorize):
            current = self._usable_keys()
            record = next((r for r in current.keys if isinstance(key_id, str) and r.id == key_id), None)
            if record is None:
                raise KeyNotFound()
            remaining = replace(current, keys=tuple(r for r in current.keys if r.id != record.id))
            if record.role == schema.ROLE_ADMIN and not remaining.has_admin(deployment_key_present):
                raise LastAdminCredential()
            state = replace(remaining, revision=self._next_keys_revision(current))
            self._store_keys(state)
            self._audit_locked("key-revoked", actor, key_id=record.id, name=record.name, role=record.role,
                               hint=record.hint)
            return record

    # --- one-time codes ----------------------------------------------------------------------

    def _load_codes(self) -> dict:
        """{"last_issued_at", "codes"}: only well-formed codes that are still usable.

        An "attempts_left" written by an earlier version is ignored (codes have no attempt
        budget any more); codes.json only ever holds codes of the last 30 minutes.
        """
        empty = {"last_issued_at": None, "codes": []}
        try:
            document = schema.parse_json_strict(self._read_bytes(self._path(CODES_FILE)))
        except FileNotFoundError:
            return empty
        except (OSError, ValueError, _Damaged) as exc:
            logger.warning("settings: codes.json is unusable (%s); no code is valid", _reason(exc))
            return empty
        if not isinstance(document, dict):
            return empty
        now = self._clock()
        last = document.get("last_issued_at")
        codes = []
        for record in document.get("codes") if isinstance(document.get("codes"), list) else []:
            if not isinstance(record, dict):
                continue
            digest, purpose = record.get("sha256"), record.get("purpose")
            issued, expires = record.get("issued_at"), record.get("expires_at")
            if (not isinstance(digest, str) or not _HEX64.fullmatch(digest) or purpose not in CODE_PURPOSES
                    or not _is_number(issued) or not _is_number(expires)):
                continue
            if expires <= now:
                continue
            codes.append({"sha256": digest, "purpose": purpose, "issued_at": issued, "expires_at": expires})
        return {"last_issued_at": last if _is_number(last) else None, "codes": codes}

    def _save_codes(self, document: Mapping) -> None:
        self._atomic_write(self._path(CODES_FILE), {"schema": FILE_SCHEMA, **document})

    @staticmethod
    def _find_code(document: Mapping, code: Any, purpose: str) -> Optional[int]:
        canonical = _canonical_code(code)
        digest = _code_digest(canonical if canonical is not None else "-" * CODE_LENGTH)
        found = None
        for index, record in enumerate(document["codes"]):
            if (record["purpose"] == purpose and hmac.compare_digest(record["sha256"], digest)
                    and canonical is not None):
                found = index
        return found

    def _refuse_code(self, document: dict, purpose: str, actor: Optional[Actor]) -> None:
        """A wrong code is recorded and changes nothing else.

        It cannot be told which code a wrong guess was aimed at, so a budget per code would be
        one budget for every caller: a few wrong guesses from anyone on the network would cancel
        the code the owner is about to type. A 100-bit code needs no budget; the gateway limits
        wrong codes per address instead (its Settings limiter).
        """
        self._audit_locked("code-refused", actor, result="refused", purpose=purpose,
                           active_codes=len(document["codes"]))

    @_write_operation
    def issue_code(self, *, purpose: str = "claim", actor: Optional[Actor] = None) -> str:
        """A new one-time code; the caller prints it to the container log and nowhere else."""
        if purpose not in CODE_PURPOSES:
            raise InvalidInput({"purpose": "Unknown purpose."})
        with self._exclusive():
            now = self._clock()
            document = self._load_codes()
            last = document["last_issued_at"]
            if last is not None and 0 <= now - last < CODE_MIN_INTERVAL_S:
                raise RateLimited(math.ceil(CODE_MIN_INTERVAL_S - (now - last)),
                                  "A code was printed less than a minute ago; use that one or wait.")
            if len(document["codes"]) >= CODE_MAX_ACTIVE:
                soonest = min(record["expires_at"] for record in document["codes"])
                raise RateLimited(math.ceil(soonest - now),
                                  f"{CODE_MAX_ACTIVE} codes are still valid; use one of them from the log.")
            canonical = "".join(secrets.choice(CODE_ALPHABET) for _ in range(CODE_LENGTH))
            document["codes"].append({"sha256": _code_digest(canonical), "purpose": purpose, "issued_at": now,
                                      "expires_at": now + CODE_TTL_S})
            document["last_issued_at"] = now
            self._save_codes(document)
            self._audit_locked("code-issued", actor, purpose=purpose, expires_at=_iso(now + CODE_TTL_S))
        return _format_code(canonical)

    @_write_operation
    def redeem_code(self, code: Any, *, purpose: str = "claim", actor: Optional[Actor] = None) -> bool:
        """Use a code up: True once for the right one; a wrong one changes nothing (it is audited)."""
        with self._exclusive():
            document = self._load_codes()
            index = self._find_code(document, code, purpose)
            if index is None:
                self._refuse_code(document, purpose, actor)
                return False
            del document["codes"][index]
            self._save_codes(document)
            self._audit_locked("code-used", actor, purpose=purpose)
            return True

    @_write_operation
    def claim(self, *, code: Any, name: str, key: str, require_key: Optional[bool] = None,
              actor: Actor, claimed: Optional[Callable[[KeysState], bool]] = None) -> KeyRecord:
        """With a one-time code: create an admin page key (and optionally require keys).

        Also the recovery path: on a damaged keys.json it starts from an empty file in the
        fail-closed state (a key required, the YAML key a client) and keeps the damaged one as
        keys.json.damaged. The key limit does not apply here, so a full list cannot block recovery.
        Whether keys.json is damaged is decided by the read just made under the lock, never by
        what this process cached (a read that may pass refuses the claim instead: Unreadable).

        `claimed(keys)` says whether the server had an admin credential before; the audit line
        is "recovery" when it had (someone who can read the log made an extra admin key) or when
        keys.json was damaged, and "claim" otherwise.
        """
        clean_name, _role = self._checked_key_fields(name, key)
        if require_key is not None and not isinstance(require_key, bool):
            raise InvalidInput({"require_key": "Expected true or false."})
        with self._exclusive():
            now = self._clock()
            codes = self._load_codes()
            index = self._find_code(codes, code, "claim")
            if index is None:
                self._refuse_code(codes, "claim", actor)
                raise InvalidCode()
            current = self._state.keys
            recovering = current.fail_closed
            was_claimed = bool(claimed(current)) if claimed is not None else current.has_admin(False)
            existing = () if recovering else current.keys
            digest = key_digest(key)
            self._check_new_key(existing, clean_name, digest)
            record = KeyRecord(id=self._new_key_id(existing), name=clean_name, role=schema.ROLE_ADMIN,
                               sha256=digest, hint=schema.key_hint(key), created_at=_iso(now))
            state = KeysState(
                revision=self._next_keys_revision(current),
                require_key=current.require_key if require_key is None else require_key,
                deployment_key_role=current.deployment_key_role, keys=existing + (record,))
            del codes["codes"][index]
            self._save_codes(codes)
            if recovering:
                self._keep_damaged_copy(KEYS_FILE)
            self._store_keys(state)
            self._audit_locked("recovery" if recovering or was_claimed else "claim", actor, key_id=record.id,
                               name=record.name, role=record.role, hint=record.hint, require_key=state.require_key,
                               recovered=recovering, server_was_claimed=was_claimed)
            return record
