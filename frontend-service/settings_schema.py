"""What the Settings page may change, and how each value is checked (stdlib only).

The gateway and settings_store.py import this module. It holds no state, reads no
environment and touches no file, except that the first wildcard host name checked
reads the vendored Public Suffix List (public_suffix.py).

Keys (phase 1), in page order
    gateway.json, the 16 preference keys (named like the app YAML's variables):
        access   TRUSTED_HOSTS, TRUSTED_ORIGINS, TRUST_PROXY_HEADERS, ALLOWED_ORIGINS
        engines  ENABLE_CANARY_ASR, ENABLE_PARAKEET_ASR, ENABLE_CHATTERBOX_TTS,
                 ENABLE_MAGPIE_TTS, ENABLE_WHISPER_CPP, ENABLE_TRAINING,
                 DEFAULT_TTS_PROVIDER, DEFAULT_STT_PROVIDER
        limits   MAX_UPLOAD_MB, MAX_TTS_CHARS, MAX_CONCURRENT_UPLOADS, MAX_CONCURRENT_FFMPEG
    keys.json, the 2 access keys (group api_access):
        require_key, deployment_key_role

    SETTINGS (tuple of Setting), BY_KEY, PREFERENCE_KEYS, ACCESS_KEYS, GUARD_KEYS,
    COMMIT_CONFIRM_KEYS, ENGINE_FLAGS, OPTIONAL_ENGINE_FLAGS, BUILTIN_ENGINES, GROUPS.
    A Setting carries kind, group, label, help, default (the gateway's own default when
    the app YAML says nothing), bounds (minimum/maximum, max_items, choices) and the flags
    guard_relevant (the value is part of the request guard: a change rebuilds the access
    policy and is run through the lockout self-check) and commit_confirm (a change is
    undone unless confirmed within 60 s).

Normalized values: bool, int, int-or-float (MAX_UPLOAD_MB), str, and tuple[str, ...] for
the host and origin lists. JSON writes tuples as lists.

API
    parse_value(key, raw) -> value
        Strict check of one value; raises ValueInvalid(message). The store uses it to
        re-check what it writes and to drop invalid hand edits when it reads.
    validate_changes(changes, *, current=None, engine_status=None, allowed=None) -> Validation
        A PUT body's `set`: .values {key: normalized}, .errors {key: message},
        .warnings (SettingWarning, need an acknowledgement), .notes {key: (text, ...)}
        (e.g. "not needed" host names, no acknowledgement). `current` (the effective
        values) limits warnings to values that change; `engine_status` maps provider id
        -> ENGINE_RUNNING / ENGINE_NOT_REACHABLE / ENGINE_NOT_INSTALLED, and turning an
        optional engine on without ENGINE_RUNNING warns. `allowed` defaults to
        PREFERENCE_KEYS.
    unacknowledged(warnings, acknowledge) -> tuple[SettingWarning, ...]
        The warnings whose token ("KEY:warning_id") is not in `acknowledge`.
    check_engine_defaults(effective, offered) -> {key: message}
        Both default engines offered and of the right kind; `offered` maps provider id ->
        kind ("tts", "stt", "training"), e.g. from offered_engines() or the real registry.
    offered_engines(effective, extra=None) -> {provider_id: kind}
        Built-in engines, plus those whose ENABLE_* flag is on, plus `extra`.
    describe() -> list[dict]        JSON-ready description of every setting, for the page
    parse_locked_keys(raw) -> (frozenset of keys, tuple of unknown names)   SETTINGS_LOCKED_KEYS
    parse_json_strict(data) -> object
        json.loads that refuses duplicate keys, NaN/Infinity (also as 1e400) and very deep
        nesting (ValueError); a leading UTF-8 BOM is tolerated.
    same_value(a, b) -> bool        equality of normalized values (list == tuple, True != 1)

    Host and origin rules, identical to the gateway's request guard:
        parse_host_header(value), normalize_host_entry(entry), host_always_accepted(host),
        normalize_origin(entry), LOCAL_HOST_SUFFIXES
    Key management:
        validate_key_name(raw), validate_role(raw), validate_page_key(raw), key_hint(key),
        ROLES, ROLE_ADMIN, ROLE_CLIENT

Warning ids (SettingWarning.id): WARN_TTS_CHARS (MAX_TTS_CHARS above 5000),
WARN_SINGLE_LABEL_WILDCARD (`*.k2o`), WARN_PROXY_HEADERS (TRUST_PROXY_HEADERS turned on),
WARN_ENGINE_NOT_REACHABLE (an optional engine offered while it does not answer).

TRUSTED_HOSTS entries are read like the gateway reads the variable (scheme, port and path
are dropped, `*.` is kept), but an entry the gateway would drop is an error here, and so
is a bare `*` (ALLOWED_HOSTS='*' in the YAML stays the off switch), a wildcard on an IP
address and a wildcard whose base is a public suffix (`*.de`, `*.co.uk`, `*.ts.net`). A
wildcard on a single private label (`*.k2o`) is accepted with a warning, and a name the
guard accepts anyway (IP addresses, single labels, localhost, .local, .lan, .internal,
.home.arpa) is kept with a "not needed" note. Origins are http(s)://host[:port] only.
"""

from __future__ import annotations

import base64
import binascii
import ipaddress
import json
import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional
from urllib.parse import urlsplit

import public_suffix

FILE_PREFERENCES = "gateway.json"
FILE_KEYS = "keys.json"

GROUP_ACCESS = "access"
GROUP_API_ACCESS = "api_access"
GROUP_ENGINES = "engines"
GROUP_LIMITS = "limits"
GROUPS = (
    (GROUP_ACCESS, "Access"),
    (GROUP_API_ACCESS, "API access & keys"),
    (GROUP_ENGINES, "Engines"),
    (GROUP_LIMITS, "Limits"),
)

KIND_HOSTS = "hosts"
KIND_ORIGINS = "origins"
KIND_BOOL = "bool"
KIND_INT = "int"
KIND_NUMBER = "number"
KIND_PROVIDER = "provider"
KIND_CHOICE = "choice"

ROLE_ADMIN = "admin"
ROLE_CLIENT = "client"
ROLES = (ROLE_ADMIN, ROLE_CLIENT)

ENGINE_RUNNING = "running"
ENGINE_NOT_REACHABLE = "not_reachable"
ENGINE_NOT_INSTALLED = "not_installed"
ENGINE_STATES = (ENGINE_RUNNING, ENGINE_NOT_REACHABLE, ENGINE_NOT_INSTALLED)

WARN_TTS_CHARS = "tts_chars_above_backend_limit"
WARN_SINGLE_LABEL_WILDCARD = "single_label_wildcard"
WARN_PROXY_HEADERS = "proxy_headers_on"
WARN_ENGINE_NOT_REACHABLE = "engine_not_reachable"
WARNING_IDS = (WARN_TTS_CHARS, WARN_SINGLE_LABEL_WILDCARD, WARN_PROXY_HEADERS, WARN_ENGINE_NOT_REACHABLE)

# What the smallest TTS backends accept (qwen3-tts, chatterbox and magpie: MAX_TEXT_CHARS).
BACKEND_TEXT_LIMIT = 5000
MAX_HOST_ENTRIES = 64
MAX_ORIGIN_ENTRIES = 16
# Raw list length beyond which nothing is looked at (a 64 KiB body holds thousands).
_MAX_RAW_ENTRIES = 1024

PAGE_KEY_PREFIX = "tts_"
MAX_KEY_NAME = 64


class ValueInvalid(ValueError):
    """A value the Settings page must refuse; `message` is safe to show next to the field."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


@dataclass(frozen=True)
class Setting:
    key: str
    kind: str
    group: str
    label: str
    help: str
    default: Any
    file: str = FILE_PREFERENCES
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    max_items: Optional[int] = None
    choices: tuple[str, ...] = ()
    guard_relevant: bool = False
    commit_confirm: bool = False
    engine: Optional[str] = None        # ENABLE_*: the provider id it offers
    engine_kind: Optional[str] = None   # ENABLE_*: that provider's kind; DEFAULT_*: the kind it names
    warn_above: Optional[float] = None
    apply: str = "live"


def _engine_flag(key: str, provider: str, kind: str, name: str, extra: str = "") -> Setting:
    return Setting(
        key=key, kind=KIND_BOOL, group=GROUP_ENGINES, label=f"Offer {name}",
        help=(f"Show {name}{extra} in the web UI and the API. Its container is installed separately "
              "(TrueNAS: delete its profiles: line in the app YAML; compose: --profile)."),
        default=False, engine=provider, engine_kind=kind)


SETTINGS: tuple[Setting, ...] = (
    Setting(
        key="TRUSTED_HOSTS", kind=KIND_HOSTS, group=GROUP_ACCESS,
        label="Host names of this server",
        help=("The names you type in the browser to reach this server, one per line (for example "
              "tts.example.com). "
              "*.example.com covers every name under example.com. IP addresses, localhost, "
              "single-label names and names under .local, .lan, .internal or .home.arpa always work "
              "and need no entry. The name you are saving from has to stay accepted."),
        default=(), max_items=MAX_HOST_ENTRIES, guard_relevant=True),
    Setting(
        key="TRUSTED_ORIGINS", kind=KIND_ORIGINS, group=GROUP_ACCESS,
        label="Public URL of this UI behind a proxy that rewrites Host",
        help=("The address a reverse proxy serves this UI under when it does not pass the Host header "
              "on, as http(s)://host[:port] without a path. Its host name is trusted too. A change has "
              "to be confirmed within 60 seconds, or it is undone."),
        default=(), max_items=MAX_ORIGIN_ENTRIES, guard_relevant=True, commit_confirm=True),
    Setting(
        key="TRUST_PROXY_HEADERS", kind=KIND_BOOL, group=GROUP_ACCESS,
        label="Believe X-Forwarded-Host (only behind a proxy you run that overwrites it)",
        help=("Use the host name a reverse proxy reports in X-Forwarded-Host instead of the Host "
              "header. Anyone who reaches the port directly can send that header, so leave it off "
              "unless every request comes through your proxy. A change has to be confirmed within "
              "60 seconds, or it is undone."),
        default=False, guard_relevant=True, commit_confirm=True),
    Setting(
        key="ALLOWED_ORIGINS", kind=KIND_ORIGINS, group=GROUP_ACCESS,
        label="Other web pages that may call the API (CORS)",
        help=("Origins, as http(s)://host[:port], of other web pages that may call this server's API "
              "from a browser. Leave it empty unless such a page exists. '*' (any page) can only be "
              "set in the app YAML. It never applies to the Settings page."),
        default=(), max_items=MAX_ORIGIN_ENTRIES, guard_relevant=True),
    Setting(
        key="require_key", kind=KIND_BOOL, group=GROUP_API_ACCESS, file=FILE_KEYS,
        label="API access",
        help=("Open on the network, or require a key on /v1, on every changing /api call and on the "
              "live microphone. Always required when the app YAML sets API_KEY. Requiring a key "
              "needs at least one key that works."),
        default=False, guard_relevant=True),
    Setting(
        key="deployment_key_role", kind=KIND_CHOICE, group=GROUP_API_ACCESS, file=FILE_KEYS,
        label="The key from the app YAML may",
        help=("admin: use the API and change settings. client: use the API only. Switching to client "
              "needs an admin key created here, and has to be done with that key, not with the YAML key."),
        default=ROLE_ADMIN, choices=ROLES, guard_relevant=True),
    _engine_flag("ENABLE_CANARY_ASR", "canary", "stt", "Canary-ASR"),
    _engine_flag("ENABLE_PARAKEET_ASR", "parakeet", "stt", "Parakeet-ASR"),
    _engine_flag("ENABLE_CHATTERBOX_TTS", "chatterbox", "tts", "Chatterbox-TTS"),
    _engine_flag("ENABLE_MAGPIE_TTS", "magpie", "tts", "Magpie-TTS"),
    _engine_flag("ENABLE_WHISPER_CPP", "whisper-cpp", "stt", "whisper.cpp", " (CPU speech-to-text)"),
    Setting(
        key="ENABLE_TRAINING", kind=KIND_BOOL, group=GROUP_ENGINES,
        label="Offer voice training",
        help=("Show the Voice Training tab and the training API. Turn it off on installs without the "
              "training container; the tab disappears when the web UI is reloaded."),
        default=True, engine="piper-training", engine_kind="training"),
    Setting(
        key="DEFAULT_TTS_PROVIDER", kind=KIND_PROVIDER, group=GROUP_ENGINES,
        label="Default text-to-speech engine",
        help="Preselected in the web UI and used by /v1/audio/speech. It has to be an offered text-to-speech engine.",
        default="piper", engine_kind="tts"),
    Setting(
        key="DEFAULT_STT_PROVIDER", kind=KIND_PROVIDER, group=GROUP_ENGINES,
        label="Default speech-to-text engine",
        help=("Preselected in the web UI and used by /v1/audio/transcriptions. It has to be an offered "
              "speech-to-text engine."),
        default="whisper", engine_kind="stt"),
    Setting(
        key="MAX_UPLOAD_MB", kind=KIND_NUMBER, group=GROUP_LIMITS,
        label="Largest upload through the gateway (MB)",
        help=("Bigger request bodies are refused with 413. Every upload is held in memory while it is "
              "forwarded, so workers x uploads at once x this size is the worst-case memory use."),
        default=512, minimum=1, maximum=2048),
    Setting(
        key="MAX_TTS_CHARS", kind=KIND_INT, group=GROUP_LIMITS,
        label="Longest text for speech (characters)",
        help=("Longer texts are refused with 422. Qwen3-TTS, Chatterbox and Magpie refuse more than "
              f"{BACKEND_TEXT_LIMIT} characters themselves."),
        default=BACKEND_TEXT_LIMIT, minimum=1, maximum=100000, warn_above=BACKEND_TEXT_LIMIT),
    Setting(
        key="MAX_CONCURRENT_UPLOADS", kind=KIND_INT, group=GROUP_LIMITS,
        label="Uploads forwarded at once (per worker)",
        help="Further uploads are answered 503 with Retry-After until one has finished.",
        default=4, minimum=1, maximum=32),
    Setting(
        key="MAX_CONCURRENT_FFMPEG", kind=KIND_INT, group=GROUP_LIMITS,
        label="Audio conversions at once (per worker)",
        help=("mp3 and resampled pcm answers of /v1/audio/speech beyond this are answered 503 with "
              "Retry-After."),
        default=4, minimum=1, maximum=32),
)

BY_KEY: Mapping[str, Setting] = MappingProxyType({setting.key: setting for setting in SETTINGS})
PREFERENCE_KEYS: tuple[str, ...] = tuple(s.key for s in SETTINGS if s.file == FILE_PREFERENCES)
ACCESS_KEYS: tuple[str, ...] = tuple(s.key for s in SETTINGS if s.file == FILE_KEYS)
GUARD_KEYS = frozenset(s.key for s in SETTINGS if s.guard_relevant)
COMMIT_CONFIRM_KEYS = frozenset(s.key for s in SETTINGS if s.commit_confirm)
# ENABLE_* key -> (provider id, kind) for every engine flag, ENABLE_TRAINING included.
ENGINE_FLAGS: Mapping[str, tuple[str, str]] = MappingProxyType({
    s.key: (s.engine, s.engine_kind) for s in SETTINGS if s.engine is not None})
# The optional engines, whose containers may be missing: offering one that does not answer warns.
OPTIONAL_ENGINE_FLAGS = frozenset(key for key in ENGINE_FLAGS if key != "ENABLE_TRAINING")
# Always offered (no flag), so a TTS and an STT engine always exist.
BUILTIN_ENGINES: Mapping[str, str] = MappingProxyType(
    {"piper": "tts", "qwen3": "tts", "whisper": "stt", "qwen3-asr": "stt"})

_ENGINE_NAMES = {
    "canary": "Canary-ASR", "parakeet": "Parakeet-ASR", "chatterbox": "Chatterbox-TTS",
    "magpie": "Magpie-TTS", "whisper-cpp": "whisper.cpp", "piper-training": "Voice training",
}
_KIND_NAMES = {"tts": "text-to-speech", "stt": "speech-to-text", "training": "training"}


# --- host names and origins: the gateway's own rules ----------------------------------------
#
# Copied from frontend-service/app.py (_LOCAL_HOST_SUFFIXES, _HOST_PATTERN, _parse_host_header,
# _normalize_host_entry, and the built-in part of _host_allowed); tests/test_settings_schema.py
# pins that both answer alike.

LOCAL_HOST_SUFFIXES = (".local", ".localhost", ".localdomain", ".lan", ".internal", ".home.arpa")
_HOST_PATTERN = re.compile(
    r"^(?:\[(?P<v6>[0-9A-Fa-f:.]+(?:%[0-9A-Za-z._~-]+)?)\]|(?P<name>[A-Za-z0-9._-]+))"
    r"(?::(?P<port>[0-9]{1,5}))?$"
)
_DNS_LABEL = re.compile(r"(?!-)[a-z0-9_-]{1,63}(?<!-)")


def parse_host_header(value: str) -> Optional[str]:
    """The lower-case hostname of a Host header, or None when it is not a plain host[:port]."""
    if not isinstance(value, str):
        return None
    match = _HOST_PATTERN.match(value.strip())
    if not match:
        return None
    host = (match.group("v6") or match.group("name")).lower().rstrip(".")
    return host or None


def normalize_host_entry(entry: str) -> Optional[str]:
    """A TRUSTED_HOSTS entry as the gateway reads it: a bare lower-case host, keeping a leading `*.`."""
    if not isinstance(entry, str):
        return None
    text = entry.strip().lower()
    if not text:
        return None
    wildcard = text.startswith("*.")
    if wildcard:
        text = text[2:]
    try:
        candidate = urlsplit(text).netloc if "://" in text else text.split("/")[0]
    except ValueError:
        return None
    parsed = parse_host_header(candidate)
    if not parsed:
        return None
    return f"*.{parsed}" if wildcard else parsed


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        return False
    return True


def host_always_accepted(host: str) -> bool:
    """Does the guard accept this (parsed) host without any TRUSTED_HOSTS entry?"""
    return _is_ip_literal(host) or "." not in host or host.endswith(LOCAL_HOST_SUFFIXES)


def _valid_dns_name(host: str) -> bool:
    return len(host) <= 253 and all(_DNS_LABEL.fullmatch(label) for label in host.split("."))


def _shown(value: Any, limit: int = 60) -> str:
    """A value quoted for a message: printable, and short."""
    text = "".join(ch if ch.isprintable() else "?" for ch in str(value).strip())
    if len(text) > limit:
        text = text[:limit] + "..."
    return f"'{text}'"


def _check_host_entry(raw: Any) -> tuple[Optional[str], Optional[str], bool, Optional[str]]:
    """(normalized, error, single-label wildcard, note) for one TRUSTED_HOSTS entry."""
    if not isinstance(raw, str):
        return None, "Every entry has to be text.", False, None
    text = raw.strip()
    shown = _shown(text)
    if text == "*":
        return None, ("A bare * is not allowed here. To switch the host check off, set "
                      "ALLOWED_HOSTS='*' in the app YAML."), False, None
    normalized = normalize_host_entry(text)
    if normalized is None:
        return None, f"{shown} is not a host name.", False, None
    wildcard = normalized.startswith("*.")
    host = normalized[2:] if wildcard else normalized
    if _is_ip_literal(host):
        if wildcard:
            return None, f"{shown}: a wildcard needs a domain name, not an IP address.", False, None
        # An IPv6 address keeps its brackets, so that the stored entry reads back the same.
        stored = f"[{host}]" if ":" in host else host
        return stored, None, False, f"{host} is an IP address; IP addresses always work and need no entry."
    if not _valid_dns_name(host):
        return None, f"{shown} is not a host name.", False, None
    if wildcard:
        if ("." + host).endswith(LOCAL_HOST_SUFFIXES):
            return normalized, None, False, f"{normalized} is not needed: these names always work."
        try:
            public = public_suffix.is_public_suffix(host)
        except public_suffix.ListUnavailable:
            return None, (f"{shown} cannot be checked right now (the public suffix list is missing); "
                          "list the names one by one."), False, None
        if public:
            return None, (f"{shown} would trust every name under {host}, which anyone can register. "
                          "List the names you use instead."), False, None
        return normalized, None, "." not in host, None
    if host_always_accepted(host):
        return normalized, None, False, f"{host} is not needed: it always works."
    return normalized, None, False, None


def normalize_origin(raw: Any, *, field: str = "TRUSTED_ORIGINS") -> str:
    """http(s)://host[:port], lower-case, without a default port; raises ValueInvalid."""
    if not isinstance(raw, str):
        raise ValueInvalid("Every entry has to be text.")
    text = raw.strip()
    shown = _shown(text)
    if text == "*":
        if field == "ALLOWED_ORIGINS":
            raise ValueInvalid("'*' (any web page) can only be set in the app YAML.")
        raise ValueInvalid("'*' is not an origin; list the address, e.g. https://tts.example.com.")
    if text.lower() == "null":
        raise ValueInvalid("'null' is not an origin.")
    if any(ch.isspace() for ch in text) or "\\" in text or "?" in text or "#" in text:
        raise ValueInvalid(f"{shown} is not an origin: use http(s)://host[:port].")
    try:
        parts = urlsplit(text)
        port = parts.port
    except ValueError:
        raise ValueInvalid(f"{shown} is not an origin: use http(s)://host[:port].") from None
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueInvalid(f"{shown}: only http:// and https:// addresses.")
    if "@" in parts.netloc:
        raise ValueInvalid(f"{shown}: an origin has no user name or password.")
    if parts.path not in ("", "/"):
        raise ValueInvalid(f"{shown}: an origin has no path.")
    host = parse_host_header(parts.netloc)
    if host is None or "%" in host or not (_is_ip_literal(host) or _valid_dns_name(host)):
        raise ValueInvalid(f"{shown} is not an origin: use http(s)://host[:port].")
    if port == 0:
        raise ValueInvalid(f"{shown}: port 0 is not a port.")
    default_port = 443 if scheme == "https" else 80
    netloc = f"[{host}]" if ":" in host else host
    if port is not None and port != default_port:
        netloc = f"{netloc}:{port}"
    return f"{scheme}://{netloc}"


# --- parsing one value --------------------------------------------------------------------

@dataclass(frozen=True)
class _Parsed:
    value: Any
    notes: tuple[str, ...] = ()
    single_label_wildcards: tuple[str, ...] = ()


def _joined(errors: list[str]) -> str:
    shown = errors[:5]
    more = f" ({len(errors) - 5} more)" if len(errors) > 5 else ""
    return " ".join(shown) + more


def _entries(raw: Any, what: str) -> list:
    if not isinstance(raw, (list, tuple)):
        raise ValueInvalid(f"Expected a list of {what}.")
    if len(raw) > _MAX_RAW_ENTRIES:
        raise ValueInvalid(f"Too many {what}.")
    # Blank lines of the text box mean nothing.
    return [entry for entry in raw if not (isinstance(entry, str) and not entry.strip())]


def _parse_hosts(setting: Setting, raw: Any) -> _Parsed:
    values: list[str] = []
    errors: list[str] = []
    notes: list[str] = []
    single: list[str] = []
    for entry in _entries(raw, "host names"):
        value, error, single_label, note = _check_host_entry(entry)
        if error:
            errors.append(error)
            continue
        if value in values:
            continue
        values.append(value)
        if single_label:
            single.append(value)
        if note:
            notes.append(note)
    if errors:
        raise ValueInvalid(_joined(errors))
    if len(values) > setting.max_items:
        raise ValueInvalid(f"At most {setting.max_items} host names.")
    return _Parsed(tuple(values), tuple(notes), tuple(single))


def _parse_origins(setting: Setting, raw: Any) -> _Parsed:
    values: list[str] = []
    errors: list[str] = []
    for entry in _entries(raw, "origins"):
        try:
            value = normalize_origin(entry, field=setting.key)
        except ValueInvalid as exc:
            errors.append(exc.message)
            continue
        if value not in values:
            values.append(value)
    if errors:
        raise ValueInvalid(_joined(errors))
    if len(values) > setting.max_items:
        raise ValueInvalid(f"At most {setting.max_items} origins.")
    return _Parsed(tuple(values))


def _bound(number: float) -> str:
    return f"{number:g}"


def _parse_number(setting: Setting, raw: Any, *, whole: bool) -> _Parsed:
    expected = "a whole number" if whole else "a number"
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueInvalid(f"Expected {expected}.")
    if isinstance(raw, float):
        if not math.isfinite(raw):
            raise ValueInvalid(f"Expected {expected}.")
        if raw.is_integer():
            raw = int(raw)
        elif whole:
            raise ValueInvalid(f"Expected {expected}.")
    if not setting.minimum <= raw <= setting.maximum:
        raise ValueInvalid(
            f"Expected {expected} from {_bound(setting.minimum)} to {_bound(setting.maximum)}.")
    return _Parsed(raw)


_PROVIDER_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")


def _parse(setting: Setting, raw: Any) -> _Parsed:
    kind = setting.kind
    if kind == KIND_HOSTS:
        return _parse_hosts(setting, raw)
    if kind == KIND_ORIGINS:
        return _parse_origins(setting, raw)
    if kind == KIND_BOOL:
        if raw is True or raw is False:
            return _Parsed(raw)
        raise ValueInvalid("Expected true or false.")
    if kind == KIND_INT:
        return _parse_number(setting, raw, whole=True)
    if kind == KIND_NUMBER:
        return _parse_number(setting, raw, whole=False)
    if kind == KIND_PROVIDER:
        if isinstance(raw, str) and _PROVIDER_ID.fullmatch(raw):
            return _Parsed(raw)
        raise ValueInvalid("Expected an engine id such as piper or whisper.")
    if kind == KIND_CHOICE:
        if isinstance(raw, str) and raw in setting.choices:
            return _Parsed(raw)
        raise ValueInvalid(f"Expected one of: {', '.join(setting.choices)}.")
    raise ValueInvalid("This setting cannot be changed.")  # pragma: no cover - every kind is handled


def parse_value(key: str, raw: Any) -> Any:
    """The normalized value of one setting; raises ValueInvalid (also for an unknown key)."""
    setting = BY_KEY.get(key) if isinstance(key, str) else None
    if setting is None:
        raise ValueInvalid("Unknown setting.")
    return _parse(setting, raw).value


def same_value(a: Any, b: Any) -> bool:
    """Equality of two normalized values: a list equals the tuple with the same items, True is not 1."""
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return tuple(a) == tuple(b)
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    return a == b


# --- validating a change set --------------------------------------------------------------

@dataclass(frozen=True)
class SettingWarning:
    """A consequence the admin has to acknowledge before the server saves."""

    id: str
    key: str
    message: str

    @property
    def token(self) -> str:
        """What `acknowledge` has to contain: "KEY:warning_id"."""
        return f"{self.key}:{self.id}"

    def as_dict(self) -> dict:
        return {"id": self.id, "key": self.key, "message": self.message, "token": self.token}


@dataclass(frozen=True)
class Validation:
    values: Mapping[str, Any]
    errors: Mapping[str, str]
    warnings: tuple[SettingWarning, ...]
    notes: Mapping[str, tuple[str, ...]]

    @property
    def ok(self) -> bool:
        return not self.errors


_UNKNOWN = object()


def _engine_warning(setting: Setting, status: Optional[str]) -> SettingWarning:
    name = _ENGINE_NAMES.get(setting.engine, setting.engine)
    if status == ENGINE_NOT_INSTALLED:
        detail = f"{name} is not installed: its container does not exist."
    elif status == ENGINE_NOT_REACHABLE:
        detail = f"{name} does not answer right now."
    else:
        detail = f"Whether {name} runs could not be checked."
    return SettingWarning(
        WARN_ENGINE_NOT_REACHABLE, setting.key,
        f"{detail} It will be offered, but shown as unavailable until its container runs.")


def _warnings_for(setting: Setting, parsed: _Parsed, before: Any,
                  engine_status: Optional[Mapping[str, str]]) -> list[SettingWarning]:
    value = parsed.value
    if before is not _UNKNOWN and same_value(before, value):
        return []
    out: list[SettingWarning] = []
    if setting.warn_above is not None and value > setting.warn_above:
        out.append(SettingWarning(
            WARN_TTS_CHARS, setting.key,
            f"Qwen3-TTS, Chatterbox and Magpie refuse more than {BACKEND_TEXT_LIMIT} characters, so a "
            f"longer text fails at the engine instead of with a clear error here."))
    if parsed.single_label_wildcards:
        earlier = set(before) if isinstance(before, (list, tuple, set, frozenset)) else set()
        new = [w for w in parsed.single_label_wildcards if w not in earlier]
        if new:
            out.append(SettingWarning(
                WARN_SINGLE_LABEL_WILDCARD, setting.key,
                f"{', '.join(new)} trusts every name under a single private label. That is fine for "
                "names your own DNS gives out, and unsafe if anyone else can create such names."))
    if setting.key == "TRUST_PROXY_HEADERS" and value is True:
        out.append(SettingWarning(
            WARN_PROXY_HEADERS, setting.key,
            "Only behind a reverse proxy you run that overwrites X-Forwarded-Host: anyone who "
            "reaches the port directly could otherwise pick the host name this server believes."))
    if setting.key in OPTIONAL_ENGINE_FLAGS and value is True:
        status = (engine_status or {}).get(setting.engine)
        if status != ENGINE_RUNNING:
            out.append(_engine_warning(setting, status))
    return out


def validate_changes(changes: Mapping[str, Any], *, current: Optional[Mapping[str, Any]] = None,
                     engine_status: Optional[Mapping[str, str]] = None,
                     allowed: Optional[Iterable[str]] = None) -> Validation:
    """Check every value of a change set on its own; see the module docstring."""
    if not isinstance(changes, Mapping):
        raise TypeError("changes must be a mapping")
    allowed_keys = frozenset(PREFERENCE_KEYS if allowed is None else allowed)
    values: dict[str, Any] = {}
    errors: dict[str, str] = {}
    warnings: list[SettingWarning] = []
    notes: dict[str, tuple[str, ...]] = {}
    for key, raw in changes.items():
        setting = BY_KEY.get(key) if isinstance(key, str) else None
        if setting is None:
            errors[str(key)[:64]] = "Unknown setting."
            continue
        if key not in allowed_keys:
            errors[key] = "This setting is not changed here."
            continue
        try:
            parsed = _parse(setting, raw)
        except ValueInvalid as exc:
            errors[key] = exc.message
            continue
        values[key] = parsed.value
        if parsed.notes:
            notes[key] = parsed.notes
        before = _UNKNOWN if current is None else current.get(key, _UNKNOWN)
        warnings.extend(_warnings_for(setting, parsed, before, engine_status))
    return Validation(values, errors, tuple(warnings), notes)


def unacknowledged(warnings: Iterable[SettingWarning], acknowledge: Optional[Iterable[str]]) -> tuple[SettingWarning, ...]:
    """The warnings whose token is not in `acknowledge` (non-strings in it are ignored)."""
    acknowledged = {item for item in (acknowledge or ()) if isinstance(item, str)}
    return tuple(warning for warning in warnings if warning.token not in acknowledged)


def offered_engines(effective: Mapping[str, Any], extra: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    """{provider id: kind} of what the registry offers for these effective values."""
    offered = dict(BUILTIN_ENGINES)
    for key, (provider, kind) in ENGINE_FLAGS.items():
        if effective.get(key, BY_KEY[key].default) is True:
            offered[provider] = kind
    if extra:
        offered.update(extra)
    return offered


def check_engine_defaults(effective: Mapping[str, Any], offered: Mapping[str, str]) -> dict[str, str]:
    """{DEFAULT_*: message} for a default engine that is not offered or of the wrong kind."""
    errors: dict[str, str] = {}
    for key in ("DEFAULT_TTS_PROVIDER", "DEFAULT_STT_PROVIDER"):
        setting = BY_KEY[key]
        provider = effective.get(key, setting.default)
        kind_name = _KIND_NAMES[setting.engine_kind]
        kind = offered.get(provider) if isinstance(provider, str) else None
        if kind is None:
            errors[key] = (f"{_shown(provider)} is not offered. Offer it first, or choose another "
                           f"default {kind_name} engine.")
        elif kind != setting.engine_kind:
            errors[key] = f"{_shown(provider)} is not a {kind_name} engine."
    return errors


def describe() -> list[dict]:
    """Every setting as a JSON-ready dict, in page order."""
    out = []
    for s in SETTINGS:
        out.append({
            "key": s.key, "kind": s.kind, "group": s.group, "label": s.label, "help": s.help,
            "file": s.file,
            "default": list(s.default) if isinstance(s.default, tuple) else s.default,
            "min": s.minimum, "max": s.maximum, "max_items": s.max_items,
            "choices": list(s.choices) or None,
            "guard_relevant": s.guard_relevant, "commit_confirm": s.commit_confirm,
            "engine": s.engine, "engine_kind": s.engine_kind, "warn_above": s.warn_above,
            "apply": s.apply,
        })
    return out


def parse_locked_keys(raw: Optional[str]) -> tuple[frozenset, tuple[str, ...]]:
    """SETTINGS_LOCKED_KEYS ("TRUSTED_HOSTS, max_tts_chars"): (keys, names that are not settings)."""
    lookup = {key.lower(): key for key in BY_KEY}
    locked: set[str] = set()
    unknown: list[str] = []
    for name in re.split(r"[\s,]+", raw or ""):
        if not name:
            continue
        key = lookup.get(name.lower())
        if key is None:
            unknown.append(name)
        else:
            locked.add(key)
    return frozenset(locked), tuple(unknown)


# --- strict JSON --------------------------------------------------------------------------

class DuplicateKeyError(ValueError):
    """A JSON object names the same key twice."""


def _unique_pairs(pairs: list) -> dict:
    out: dict = {}
    for key, value in pairs:
        if key in out:
            raise DuplicateKeyError(f"duplicate key {_shown(key, 64)}")
        out[key] = value
    return out


def _no_constant(name: str):
    raise ValueError(f"{name} is not allowed")


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"{text[:20]} is out of range")
    return value


def parse_json_strict(data: Any) -> Any:
    """json.loads refusing duplicate keys, NaN/Infinity and very deep nesting (ValueError)."""
    if isinstance(data, (bytes, bytearray, memoryview)):
        text = bytes(data).decode("utf-8")
    elif isinstance(data, str):
        text = data
    else:
        raise TypeError("expected bytes or str")
    if text.startswith("\ufeff"):
        text = text[1:]
    try:
        return json.loads(text, object_pairs_hook=_unique_pairs, parse_constant=_no_constant,
                          parse_float=_finite_float)
    except RecursionError:
        raise ValueError("nested too deeply") from None


# --- page keys ----------------------------------------------------------------------------

_PAGE_KEY = re.compile(r"tts_[A-Za-z0-9_-]{43}")


def validate_key_name(raw: Any) -> str:
    """1-64 printable characters, surrounding whitespace removed; raises ValueInvalid."""
    if not isinstance(raw, str):
        raise ValueInvalid("Expected a name.")
    name = raw.strip()
    if not 1 <= len(name) <= MAX_KEY_NAME:
        raise ValueInvalid(f"A name of 1 to {MAX_KEY_NAME} characters.")
    if not name.isprintable():
        raise ValueInvalid("The name may only contain printable characters.")
    return name


def validate_role(raw: Any) -> str:
    if isinstance(raw, str) and raw in ROLES:
        return raw
    raise ValueInvalid(f"Expected one of: {', '.join(ROLES)}.")


def validate_page_key(raw: Any) -> str:
    """A key the page generated: tts_ + 43 base64url characters (256 bits). Never echoed back."""
    message = "Expected a key generated by the Settings page (tts_ followed by 43 characters)."
    if not isinstance(raw, str) or not _PAGE_KEY.fullmatch(raw):
        raise ValueInvalid(message)
    body = raw[len(PAGE_KEY_PREFIX):]
    try:
        decoded = base64.urlsafe_b64decode(body + "=")
    except (binascii.Error, ValueError):
        raise ValueInvalid(message) from None
    # 43 characters carry 258 bits; the canonical encoding of 32 bytes leaves the last 2 zero.
    if len(decoded) != 32 or base64.urlsafe_b64encode(decoded).decode("ascii").rstrip("=") != body:
        raise ValueInvalid(message)
    return raw


def key_hint(key: str) -> str:
    """What is shown of a page key after it is created: its prefix and first 4 characters."""
    return key[: len(PAGE_KEY_PREFIX) + 4]
