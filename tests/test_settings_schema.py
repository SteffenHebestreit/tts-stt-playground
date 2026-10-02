"""settings_schema.py and public_suffix.py: what the Settings page accepts, refuses and warns about.

The validators are the only thing between a form field and the gateway's request guard, so
the host-name rules are pinned here twice: against the cases the design names (`*.de`,
`*.co.uk`, `*.ts.net` and a bare `*` refused, `*.k2o` accepted with a warning), and against
the gateway's own parser in app.py, which they must read exactly like.
"""

from __future__ import annotations

import ast
import base64
import importlib
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from frontend_loader import SERVICE_DIR, load_frontend_app


def _import(name: str):
    sys.path.insert(0, str(SERVICE_DIR))
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(SERVICE_DIR))


public_suffix = _import("public_suffix")
schema = _import("settings_schema")


def _hosts(*entries, current=None):
    return schema.validate_changes({"TRUSTED_HOSTS": list(entries)}, current=current)


def _page_key(raw: bytes = None) -> str:
    raw = os.urandom(32) if raw is None else raw
    return "tts_" + base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


# --- the public suffix matcher --------------------------------------------------------------

SYNTHETIC_LIST = [
    "// comments and blank lines are not rules",
    "",
    "com",
    "co.uk",
    "*.ck",
    "!www.ck",
    "*.kawasaki.jp",
    "!city.kawasaki.jp",
    "\u516c\u53f8.cn",
    "github.io   anything after the first whitespace is ignored",
    "*.bad*.example",
]


@pytest.fixture(scope="module")
def synthetic():
    return public_suffix.PublicSuffixList.from_lines(SYNTHETIC_LIST)


@pytest.mark.parametrize("name,expected", [
    ("com", True), ("example.com", False), ("COM.", True),
    ("co.uk", True), ("uk", True), ("example.co.uk", False),
    # a wildcard rule: every name directly under ck is public, ck itself is a known top label
    ("ck", True), ("foo.ck", True), ("bar.foo.ck", False),
    # an exception rule: www.ck is registrable, and so is everything under it
    ("www.ck", False), ("a.www.ck", False),
    ("x.kawasaki.jp", True), ("city.kawasaki.jp", False), ("kawasaki.jp", False),
    # internationalised rules match in either spelling
    ("xn--55qx5d.cn", True), ("\u516c\u53f8.cn", True),
    ("github.io", True), ("user.github.io", False),
    # a single label the list does not know is a private name, not a public suffix
    ("k2o", False), ("speach.k2o", False),
    ("", False), (".", False), ("a..b", False),
])
def test_the_matcher_follows_the_format(synthetic, name, expected):
    assert synthetic.is_public_suffix(name) is expected


def test_a_rule_with_an_inner_wildcard_is_skipped(synthetic):
    assert not synthetic.is_public_suffix("x.bad1.example")
    # 4 normal rules, 2 wildcards, 2 exceptions; the inner-wildcard line is not one of them
    assert len(synthetic) == 8


VENDORED = SERVICE_DIR / "public_suffix_list.dat"


def test_the_vendored_list_keeps_its_licence_header_and_both_sections():
    text = VENDORED.read_text(encoding="utf-8")
    assert text.startswith(
        "// This Source Code Form is subject to the terms of the Mozilla Public\n"
        "// License, v. 2.0. If a copy of the MPL was not distributed with this\n"
        "// file, You can obtain one at https://mozilla.org/MPL/2.0/.")
    assert "// VERSION:" in text
    assert "===BEGIN ICANN DOMAINS===" in text and "===BEGIN PRIVATE DOMAINS===" in text
    assert len(public_suffix.default_list()) > 9000


@pytest.mark.parametrize("name,expected", [
    ("de", True), ("com", True), ("co.uk", True), ("ts.net", True), ("github.io", True),
    ("home.arpa", True), ("foo.ck", True), ("ck", True),
    ("k2o", False), ("speach.k2o", False), ("example.co.uk", False), ("www.ck", False),
    ("example.com", False), ("lan", False), ("internal", False),
])
def test_the_vendored_list(name, expected):
    assert public_suffix.is_public_suffix(name) is expected


def test_without_the_list_a_wildcard_is_refused_not_waved_through(monkeypatch, tmp_path):
    monkeypatch.setattr(public_suffix, "_default", None)
    monkeypatch.setattr(public_suffix, "DEFAULT_PATH", str(tmp_path / "missing.dat"))
    with pytest.raises(public_suffix.ListUnavailable):
        public_suffix.default_list()
    result = _hosts("*.example.com", "speach.k2o")
    assert "cannot be checked" in result.errors["TRUSTED_HOSTS"]

    # A truncated file is as good as none: it would let every wildcard through.
    short = tmp_path / "short.dat"
    short.write_text("com\nnet\n", encoding="utf-8")
    monkeypatch.setattr(public_suffix, "DEFAULT_PATH", str(short))
    with pytest.raises(public_suffix.ListUnavailable):
        public_suffix.default_list()
    # Exact names need no list.
    assert _hosts("speach.k2o").ok


# --- TRUSTED_HOSTS ---------------------------------------------------------------------------

@pytest.mark.parametrize("entry", ["*.de", "*.co.uk", "*.ts.net", "*.com", "*.github.io", "*.ck", "*.foo.ck"])
def test_a_wildcard_on_a_public_suffix_is_refused(entry):
    result = _hosts("speach.k2o", entry)
    assert not result.ok and result.values == {}
    assert "anyone can register" in result.errors["TRUSTED_HOSTS"]
    assert entry in result.errors["TRUSTED_HOSTS"]


def test_a_bare_star_is_refused_and_points_at_the_yaml_switch():
    result = _hosts("*")
    assert "ALLOWED_HOSTS='*'" in result.errors["TRUSTED_HOSTS"]


@pytest.mark.parametrize("entry", [
    "a..b", "-bad.k2o", "bad-.k2o", "ex ample.k2o", "user@host.k2o", "[::1", "*.", "*.*.k2o", "**.k2o",
    "b\u00fccher.k2o", "x" * 64 + ".k2o", ".".join(["abcdefghi"] * 26), "http://", "host:port",
    "*.192.168.1.1", "*.[::1]",
])
def test_an_entry_the_gateway_would_drop_is_a_field_error(entry):
    result = _hosts("speach.k2o", entry)
    assert set(result.errors) == {"TRUSTED_HOSTS"}
    assert result.values == {}


@pytest.mark.parametrize("raw", ["speach.k2o", None, 5, {"speach.k2o": True}, [42], [["speach.k2o"]]])
def test_the_value_has_to_be_a_list_of_text(raw):
    with pytest.raises(schema.ValueInvalid):
        schema.parse_value("TRUSTED_HOSTS", raw)


def test_a_single_label_wildcard_is_accepted_with_a_warning():
    result = _hosts("speach.k2o", "*.k2o")
    assert result.ok
    assert result.values == {"TRUSTED_HOSTS": ("speach.k2o", "*.k2o")}
    assert [w.token for w in result.warnings] == ["TRUSTED_HOSTS:single_label_wildcard"]
    assert result.warnings[0].id == schema.WARN_SINGLE_LABEL_WILDCARD
    assert "*.k2o" in result.warnings[0].message
    # Only a wildcard that is new needs the acknowledgement again.
    assert _hosts("*.k2o", "tts.k2o", current={"TRUSTED_HOSTS": ("*.k2o",)}).warnings == ()
    assert _hosts("*.k2o", current={"TRUSTED_HOSTS": ["speach.k2o"]}).warnings != ()
    # A wildcard below a registrable domain needs no warning.
    assert _hosts("*.example.com").warnings == ()


def test_entries_are_stored_the_way_the_gateway_reads_them():
    result = _hosts("Speach.K2O", "http://speach.k2o:3000/settings", "speach.k2o.", "  tts.k2o  ", "", "   ",
                    "*.Example.COM")
    assert result.values == {"TRUSTED_HOSTS": ("speach.k2o", "tts.k2o", "*.example.com")}


def test_names_that_always_work_are_kept_with_a_not_needed_note():
    entries = ["192.168.1.20", "localhost", "truenas", "nas.local", "*.lan", "*.home.arpa", "[::1]", "x.internal"]
    result = _hosts(*entries)
    assert result.ok and result.warnings == ()
    assert result.values["TRUSTED_HOSTS"] == (
        "192.168.1.20", "localhost", "truenas", "nas.local", "*.lan", "*.home.arpa", "[::1]", "x.internal")
    assert len(result.notes["TRUSTED_HOSTS"]) == len(entries)
    assert all("need" in note for note in result.notes["TRUSTED_HOSTS"])


def test_at_most_64_host_names():
    assert _hosts(*(f"h{i}.k2o" for i in range(64))).ok
    result = _hosts(*(f"h{i}.k2o" for i in range(65)))
    assert "At most 64" in result.errors["TRUSTED_HOSTS"]
    # duplicates do not count
    assert _hosts(*(["speach.k2o"] * 200)).values == {"TRUSTED_HOSTS": ("speach.k2o",)}


# --- origins ---------------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["TRUSTED_ORIGINS", "ALLOWED_ORIGINS"])
@pytest.mark.parametrize("entry", [
    "*", "null", "NULL", "ftp://tts.example.com", "https://tts.example.com/path", "https://tts.example.com/ui/",
    "https://user@tts.example.com", "https://user:pw@tts.example.com", "https://tts.example.com?x=1",
    "https://tts.example.com#top", "https://*.example.com", "https://tts.example.com:99999",
    "https://tts.example.com:0", "tts.example.com", "https://", "http://[fe80::1%25eth0]:3000",
    "https://tts example.com", "https://tts.example.com\\x", "javascript:alert(1)", "https://a..b",
])
def test_an_origin_is_scheme_host_and_port_only(key, entry):
    result = schema.validate_changes({key: ["https://ok.example.com", entry]})
    assert set(result.errors) == {key}, result


def test_a_star_origin_can_only_come_from_the_yaml():
    assert "app YAML" in schema.validate_changes({"ALLOWED_ORIGINS": ["*"]}).errors["ALLOWED_ORIGINS"]


@pytest.mark.parametrize("entry,stored", [
    ("HTTPS://Tts.Example.com:443/", "https://tts.example.com"),
    ("http://tts.example.com:80", "http://tts.example.com"),
    ("https://tts.example.com:8443", "https://tts.example.com:8443"),
    ("http://192.168.1.20:3000", "http://192.168.1.20:3000"),
    ("http://[::1]:3000", "http://[::1]:3000"),
    ("  https://speach.k2o  ", "https://speach.k2o"),
])
def test_origins_are_normalized(entry, stored):
    assert schema.parse_value("TRUSTED_ORIGINS", [entry]) == (stored,)
    assert schema.parse_value("ALLOWED_ORIGINS", [entry, stored]) == (stored,)


def test_at_most_16_origins():
    assert schema.validate_changes({"TRUSTED_ORIGINS": [f"https://h{i}.example.com" for i in range(16)]}).ok
    result = schema.validate_changes({"TRUSTED_ORIGINS": [f"https://h{i}.example.com" for i in range(17)]})
    assert "At most 16" in result.errors["TRUSTED_ORIGINS"]


# --- the other kinds and their bounds ----------------------------------------------------------

@pytest.mark.parametrize("key,accepted,refused", [
    ("MAX_UPLOAD_MB", [(1, 1), (2048, 2048), (1.5, 1.5), (512.0, 512)], [0, 0.5, 2049, -1, True, "512", None, float("nan"), float("inf")]),
    ("MAX_TTS_CHARS", [(1, 1), (100000, 100000), (5000.0, 5000)], [0, 100001, 2.5, True, "5000", None]),
    ("MAX_CONCURRENT_UPLOADS", [(1, 1), (32, 32)], [0, 33, False, 4.5]),
    ("MAX_CONCURRENT_FFMPEG", [(1, 1), (32, 32)], [0, 33, None]),
    ("TRUST_PROXY_HEADERS", [(True, True), (False, False)], [1, 0, "true", None]),
    ("ENABLE_MAGPIE_TTS", [(True, True), (False, False)], [1, "yes"]),
    ("ENABLE_TRAINING", [(False, False)], ["false"]),
    ("DEFAULT_TTS_PROVIDER", [("magpie", "magpie"), ("qwen3", "qwen3")], ["", "bad id", "x" * 65, "piper\n", 5, None]),
    ("DEFAULT_STT_PROVIDER", [("whisper-cpp", "whisper-cpp"), ("qwen3-asr", "qwen3-asr")], ["-x", ["whisper"]]),
    ("require_key", [(True, True)], ["on"]),
    ("deployment_key_role", [("admin", "admin"), ("client", "client")], ["root", "Admin", None]),
])
def test_bounds_and_types(key, accepted, refused):
    for raw, normalized in accepted:
        value = schema.parse_value(key, raw)
        assert value == normalized and type(value) is type(normalized)
    for raw in refused:
        with pytest.raises(schema.ValueInvalid):
            schema.parse_value(key, raw)


def test_an_unknown_key_is_refused():
    with pytest.raises(schema.ValueInvalid):
        schema.parse_value("API_KEY", "secret")
    result = schema.validate_changes({"API_KEY": "x", "ALLOWED_HOSTS": "*", "MAX_TTS_CHARS": 100})
    assert result.errors == {"API_KEY": "Unknown setting.", "ALLOWED_HOSTS": "Unknown setting."}
    assert result.values == {"MAX_TTS_CHARS": 100}


def test_the_access_keys_are_not_preferences():
    result = schema.validate_changes({"require_key": True})
    assert "require_key" in result.errors
    allowed = schema.validate_changes({"require_key": True, "deployment_key_role": "client"},
                                      allowed=schema.ACCESS_KEYS)
    assert allowed.ok and allowed.values == {"require_key": True, "deployment_key_role": "client"}
    with pytest.raises(TypeError):
        schema.validate_changes(["require_key"])


def test_every_value_reads_back_as_itself():
    samples = {
        "TRUSTED_HOSTS": ["Speach.K2O:3000", "*.k2o", "[::1]", "192.168.1.20", "*.home.arpa"],
        "TRUSTED_ORIGINS": ["HTTPS://Tts.Example.com:443/", "http://[::1]:3000"],
        "ALLOWED_ORIGINS": ["https://app.example.com:8443"],
        "TRUST_PROXY_HEADERS": True, "ENABLE_CANARY_ASR": True, "DEFAULT_TTS_PROVIDER": "magpie",
        "MAX_UPLOAD_MB": 1.5, "MAX_TTS_CHARS": 7000.0, "MAX_CONCURRENT_UPLOADS": 8,
    }
    for key, raw in samples.items():
        once = schema.parse_value(key, raw)
        twice = schema.parse_value(key, json.loads(json.dumps(list(once) if isinstance(once, tuple) else once)))
        assert twice == once, key


# --- warnings ------------------------------------------------------------------------------

def test_more_text_than_the_engines_take_needs_an_acknowledgement():
    result = schema.validate_changes({"MAX_TTS_CHARS": 5001})
    assert [w.token for w in result.warnings] == ["MAX_TTS_CHARS:tts_chars_above_backend_limit"]
    assert schema.validate_changes({"MAX_TTS_CHARS": 5000}).warnings == ()
    assert schema.validate_changes({"MAX_TTS_CHARS": 8000}, current={"MAX_TTS_CHARS": 8000}).warnings == ()
    assert schema.validate_changes({"MAX_TTS_CHARS": 9000}, current={"MAX_TTS_CHARS": 8000}).warnings != ()


def test_turning_proxy_headers_on_needs_an_acknowledgement():
    result = schema.validate_changes({"TRUST_PROXY_HEADERS": True})
    assert [w.id for w in result.warnings] == [schema.WARN_PROXY_HEADERS]
    assert schema.validate_changes({"TRUST_PROXY_HEADERS": True}, current={"TRUST_PROXY_HEADERS": True}).warnings == ()
    assert schema.validate_changes({"TRUST_PROXY_HEADERS": False}).warnings == ()


@pytest.mark.parametrize("status,words", [
    ({"magpie": "not_installed"}, "not installed"),
    ({"magpie": "not_reachable"}, "does not answer"),
    ({}, "could not be checked"),
    (None, "could not be checked"),
])
def test_offering_an_engine_that_does_not_answer_needs_an_acknowledgement(status, words):
    result = schema.validate_changes({"ENABLE_MAGPIE_TTS": True}, engine_status=status)
    assert [w.token for w in result.warnings] == ["ENABLE_MAGPIE_TTS:engine_not_reachable"]
    assert words in result.warnings[0].message


def test_a_running_engine_an_unchanged_flag_and_training_need_nothing():
    assert schema.validate_changes({"ENABLE_MAGPIE_TTS": True}, engine_status={"magpie": "running"}).warnings == ()
    assert schema.validate_changes({"ENABLE_MAGPIE_TTS": True}, current={"ENABLE_MAGPIE_TTS": True}).warnings == ()
    assert schema.validate_changes({"ENABLE_MAGPIE_TTS": False}).warnings == ()
    assert schema.validate_changes({"ENABLE_TRAINING": True}).warnings == ()


def test_unacknowledged():
    result = schema.validate_changes({"MAX_TTS_CHARS": 9000, "TRUST_PROXY_HEADERS": True, "ENABLE_CANARY_ASR": True},
                                     engine_status={"canary": "not_installed"})
    tokens = {w.token for w in result.warnings}
    assert tokens == {"MAX_TTS_CHARS:tts_chars_above_backend_limit", "TRUST_PROXY_HEADERS:proxy_headers_on",
                      "ENABLE_CANARY_ASR:engine_not_reachable"}
    assert schema.unacknowledged(result.warnings, tokens) == ()
    left = schema.unacknowledged(result.warnings, ["MAX_TTS_CHARS:tts_chars_above_backend_limit", 7, None])
    assert {w.key for w in left} == {"TRUST_PROXY_HEADERS", "ENABLE_CANARY_ASR"}
    # the bare warning id is not enough: an acknowledgement names the field
    assert len(schema.unacknowledged(result.warnings, list(schema.WARNING_IDS))) == 3
    assert all(set(w.as_dict()) == {"id", "key", "message", "token"} for w in result.warnings)


# --- engines -----------------------------------------------------------------------------

def test_a_default_engine_has_to_be_offered_and_of_its_kind():
    offered = schema.offered_engines({})
    assert offered == {"piper": "tts", "qwen3": "tts", "whisper": "stt", "qwen3-asr": "stt",
                       "piper-training": "training"}
    assert schema.check_engine_defaults({}, offered) == {}

    candidate = {"DEFAULT_TTS_PROVIDER": "magpie"}
    errors = schema.check_engine_defaults(candidate, schema.offered_engines(candidate))
    assert "not offered" in errors["DEFAULT_TTS_PROVIDER"]
    candidate["ENABLE_MAGPIE_TTS"] = True
    assert schema.check_engine_defaults(candidate, schema.offered_engines(candidate)) == {}

    wrong_kind = {"DEFAULT_STT_PROVIDER": "piper", "DEFAULT_TTS_PROVIDER": "whisper"}
    errors = schema.check_engine_defaults(wrong_kind, offered)
    assert "not a speech-to-text engine" in errors["DEFAULT_STT_PROVIDER"]
    assert "not a text-to-speech engine" in errors["DEFAULT_TTS_PROVIDER"]

    # engines from PROVIDER_REGISTRY_JSON count as offered
    custom = {"DEFAULT_TTS_PROVIDER": "my-tts"}
    assert schema.check_engine_defaults(custom, schema.offered_engines(custom, {"my-tts": "tts"})) == {}
    assert "piper-training" not in schema.offered_engines({"ENABLE_TRAINING": False})


def test_the_schema_names_the_engines_the_gateway_registers():
    everything = {key: "true" for key in schema.ENGINE_FLAGS}
    app = load_frontend_app(everything)
    providers = app.PROVIDER_REGISTRY["providers"]
    for key, (provider, kind) in schema.ENGINE_FLAGS.items():
        assert providers[provider]["kind"] == kind, key
    for provider, kind in schema.BUILTIN_ENGINES.items():
        assert providers[provider]["kind"] == kind, provider
    # Built-ins have no flag: they stay offered whatever the flags say.
    nothing = load_frontend_app({key: "false" for key in schema.OPTIONAL_ENGINE_FLAGS})
    assert set(schema.BUILTIN_ENGINES) <= set(nothing.PROVIDER_REGISTRY["providers"])


def test_the_defaults_are_the_gateways_own():
    app = load_frontend_app({})
    by_key = schema.BY_KEY
    assert by_key["MAX_UPLOAD_MB"].default == app.MAX_UPLOAD_MB
    assert by_key["MAX_TTS_CHARS"].default == app.MAX_TTS_CHARS
    assert by_key["MAX_CONCURRENT_UPLOADS"].default == app.MAX_CONCURRENT_UPLOADS
    assert by_key["MAX_CONCURRENT_FFMPEG"].default == app.openai_router.MAX_CONCURRENT_FFMPEG
    ui = app.PROVIDER_REGISTRY["ui"]
    assert by_key["DEFAULT_TTS_PROVIDER"].default == ui["default_tts_provider"]
    assert by_key["DEFAULT_STT_PROVIDER"].default == ui["default_stt_provider"]
    for key in schema.OPTIONAL_ENGINE_FLAGS:
        assert by_key[key].default is False and getattr(app, key) is False
    assert by_key["TRUST_PROXY_HEADERS"].default is app.TRUST_PROXY_HEADERS is False
    assert by_key["ENABLE_TRAINING"].default is True


# --- the table itself ------------------------------------------------------------------------

def test_the_phase_one_keys():
    assert schema.PREFERENCE_KEYS == (
        "TRUSTED_HOSTS", "TRUSTED_ORIGINS", "TRUST_PROXY_HEADERS", "ALLOWED_ORIGINS",
        "ENABLE_CANARY_ASR", "ENABLE_PARAKEET_ASR", "ENABLE_CHATTERBOX_TTS", "ENABLE_MAGPIE_TTS",
        "ENABLE_WHISPER_CPP", "ENABLE_TRAINING", "DEFAULT_TTS_PROVIDER", "DEFAULT_STT_PROVIDER",
        "MAX_UPLOAD_MB", "MAX_TTS_CHARS", "MAX_CONCURRENT_UPLOADS", "MAX_CONCURRENT_FFMPEG",
    )
    assert schema.ACCESS_KEYS == ("require_key", "deployment_key_role")
    assert len(schema.SETTINGS) == 18
    assert schema.GUARD_KEYS == {"TRUSTED_HOSTS", "TRUSTED_ORIGINS", "TRUST_PROXY_HEADERS", "ALLOWED_ORIGINS",
                                 "require_key", "deployment_key_role"}
    assert schema.COMMIT_CONFIRM_KEYS == {"TRUSTED_ORIGINS", "TRUST_PROXY_HEADERS"}
    assert schema.OPTIONAL_ENGINE_FLAGS == {"ENABLE_CANARY_ASR", "ENABLE_PARAKEET_ASR", "ENABLE_CHATTERBOX_TTS",
                                            "ENABLE_MAGPIE_TTS", "ENABLE_WHISPER_CPP"}
    # The switches that govern the page itself, and the YAML-only knobs, are not in it.
    for kept_out in ("ALLOWED_HOSTS", "ALLOW_CREDENTIALS", "API_KEY", "ENABLE_SETTINGS_UI", "SETTINGS_LOCKED_KEYS",
                     "PROVIDER_REGISTRY_JSON", "FRONTEND_WORKERS", "MODEL_TTL"):
        assert kept_out not in schema.BY_KEY


def test_every_setting_is_described_and_its_default_is_valid():
    groups = {group for group, _label in schema.GROUPS}
    for setting in schema.SETTINGS:
        assert setting.label and setting.help and setting.group in groups, setting.key
        assert setting.file in (schema.FILE_PREFERENCES, schema.FILE_KEYS)
        default = setting.default
        assert schema.same_value(schema.parse_value(setting.key, list(default) if isinstance(default, tuple) else default),
                                 default), setting.key
        if setting.kind in (schema.KIND_INT, schema.KIND_NUMBER):
            assert setting.minimum is not None and setting.maximum is not None
        if setting.kind in (schema.KIND_HOSTS, schema.KIND_ORIGINS):
            assert setting.max_items
    description = schema.describe()
    assert [d["key"] for d in description] == [s.key for s in schema.SETTINGS]
    json.dumps(description)  # JSON-ready
    by_key = {d["key"]: d for d in description}
    assert by_key["MAX_TTS_CHARS"]["min"] == 1 and by_key["MAX_TTS_CHARS"]["max"] == 100000
    assert by_key["MAX_TTS_CHARS"]["warn_above"] == 5000
    assert by_key["TRUSTED_HOSTS"]["max_items"] == 64 and by_key["TRUSTED_HOSTS"]["default"] == []
    assert by_key["TRUSTED_ORIGINS"]["commit_confirm"] is True and by_key["TRUSTED_ORIGINS"]["guard_relevant"] is True
    assert by_key["deployment_key_role"]["choices"] == ["admin", "client"]
    assert by_key["require_key"]["file"] == "keys.json"


def test_locked_keys_are_parsed_leniently():
    locked, unknown = schema.parse_locked_keys("TRUSTED_HOSTS, max_tts_chars\nrequire_key  bogus,,")
    assert locked == {"TRUSTED_HOSTS", "MAX_TTS_CHARS", "require_key"}
    assert unknown == ("bogus",)
    assert schema.parse_locked_keys("") == (frozenset(), ())
    assert schema.parse_locked_keys(None) == (frozenset(), ())


def test_same_value():
    assert schema.same_value(["a"], ("a",))
    assert not schema.same_value(True, 1)
    assert schema.same_value(512, 512.0)
    assert not schema.same_value(("a", "b"), ("b", "a"))


# --- strict JSON ---------------------------------------------------------------------------

_REFUSED_JSON = {
    "duplicate": '{"a": 1, "a": 2}', "nested-duplicate": '{"x": {"a": 1, "a": 1}}', "nan": "NaN",
    "infinity": '{"x": Infinity}', "minus-infinity": '{"x": -Infinity}', "overflow": "1e400",
    "deep": "[" * 100000, "truncated": "{", "empty": "", "garbage": "\u00ff\u00fe",
}


@pytest.mark.parametrize("text", list(_REFUSED_JSON.values()), ids=list(_REFUSED_JSON))
def test_strict_json_refuses(text):
    with pytest.raises(ValueError):
        schema.parse_json_strict(text)


def test_strict_json_reads_the_ordinary():
    assert schema.parse_json_strict(b'\xef\xbb\xbf{"a": [1, 2.5, "x"], "b": null}') == {"a": [1, 2.5, "x"], "b": None}
    assert schema.parse_json_strict('{"a": 1}') == {"a": 1}
    with pytest.raises(ValueError):
        schema.parse_json_strict(b'{"a": "\xff"}')
    with pytest.raises(TypeError):
        schema.parse_json_strict(5)
    assert isinstance(schema.DuplicateKeyError("x"), ValueError)


# --- page keys -----------------------------------------------------------------------------

def test_a_page_key_is_tts_and_256_bits_of_base64url():
    key = _page_key()
    assert schema.validate_page_key(key) == key and len(key) == 47
    assert schema.key_hint(key) == key[:8]
    canonical = _page_key(b"\x00" * 32)
    assert canonical.endswith("A")
    for bad in (key[:-1], key + "A", "tts-" + key[4:], "TTS_" + key[4:], key[:-1] + "+", key[:-1] + "/",
                canonical[:-1] + "B", key + "\n", 42, None, ""):
        with pytest.raises(schema.ValueInvalid) as info:
            schema.validate_page_key(bad)
        if isinstance(bad, str) and bad:
            assert bad.strip() not in info.value.message


def test_key_names_and_roles():
    assert schema.validate_key_name("  Home Assistant  ") == "Home Assistant"
    assert schema.validate_key_name("x" * 64) == "x" * 64
    for bad in ("", "   ", "x" * 65, "a\nb", "tab\there", 5, None):
        with pytest.raises(schema.ValueInvalid):
            schema.validate_key_name(bad)
    assert schema.validate_role("client") == "client"
    with pytest.raises(schema.ValueInvalid):
        schema.validate_role("owner")


# --- the same rules as the gateway ------------------------------------------------------------

HOST_ENTRIES = [
    "speach.k2o", "SPEACH.K2O", "http://speach.k2o:3000/x", "https://[::1]:3000", "*.k2o", "*.Example.com",
    "host:3000", "", "  ", "a b", "user@host", "host/path", "*.", "*", "[::1]", "[fe80::1%eth0]:80",
    "1.2.3.4:80", "x..y", "-x.k2o", "trailing.dot.", "tts.k2o\n", "*.lan", "http://user@h.k2o/",
]
HOST_HEADERS = [
    "speach.k2o:3000", "[::1]:3000", "user@host", "evil.example", "host:abc", "a b", "localhost",
    "192.168.1.20:3000", "truenas", "nas.local:3000", "x.home.arpa", "home.arpa", "[fe80::1%eth0]",
    "foo.lan.", "tts.example.com", "UPPER.LOCAL",
]


@pytest.fixture(scope="module")
def gateway():
    return load_frontend_app({})


@pytest.mark.parametrize("entry", HOST_ENTRIES)
def test_host_entries_are_read_like_the_gateway_reads_them(gateway, entry):
    assert schema.normalize_host_entry(entry) == gateway._normalize_host_entry(entry)


@pytest.mark.parametrize("header", HOST_HEADERS)
def test_host_headers_are_parsed_like_the_gateway_parses_them(gateway, header):
    host = schema.parse_host_header(header)
    assert host == gateway._parse_host_header(header)
    if host is not None:
        # With nothing configured, the guard accepts exactly what the schema calls "always works".
        assert gateway._host_allowed(header) is schema.host_always_accepted(host)


@pytest.mark.parametrize("origin", [
    "https://tts.example.com", "http://192.168.1.20:3000", "http://[::1]:3000", "HTTPS://Tts.Example.com:443/",
])
def test_an_origins_host_is_the_one_the_gateway_trusts(gateway, origin):
    assert urlsplit(schema.normalize_origin(origin)).hostname == gateway._origin_key(origin)[0]


def test_the_host_suffixes_match_the_gateway(gateway):
    assert schema.LOCAL_HOST_SUFFIXES == gateway._LOCAL_HOST_SUFFIXES
    assert schema._HOST_PATTERN.pattern == gateway._HOST_PATTERN.pattern


# --- the modules themselves --------------------------------------------------------------------

MODULES = ("public_suffix.py", "settings_schema.py", "settings_store.py")
_WRITES = {"mkdir", "makedirs", "mkdtemp", "mkstemp", "touch", "write_text", "write_bytes", "open", "replace",
           "unlink", "remove", "rename", "chmod", "getenv"}


def _module_level_calls(tree: ast.Module):
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(node, ast.Call):
            yield node
        stack.extend(ast.iter_child_nodes(node))


@pytest.mark.parametrize("name", MODULES)
def test_the_modules_need_only_the_standard_library(name):
    tree = ast.parse((SERVICE_DIR / name).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            imported.add(node.module.split(".")[0])
    local = {"public_suffix", "settings_schema"}
    foreign = sorted(module for module in imported - local if module not in sys.stdlib_module_names)
    assert not foreign, f"{name} imports {foreign}"


@pytest.mark.parametrize("name", MODULES)
def test_importing_the_modules_touches_nothing(name):
    tree = ast.parse((SERVICE_DIR / name).read_text(encoding="utf-8"))
    offenders = []
    for call in _module_level_calls(tree):
        func = call.func
        called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if called in _WRITES:
            offenders.append(f"line {call.lineno}: {ast.unparse(call)[:60]}")
    assert not offenders, offenders
    assert "environ" not in {node.attr for node in ast.walk(ast.Module(
        body=[n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.ClassDef))], type_ignores=[]))
        if isinstance(node, ast.Attribute)}


def test_the_new_files_are_where_the_image_will_look_for_them():
    for name in (*MODULES, "public_suffix_list.dat"):
        assert (SERVICE_DIR / name).is_file(), name
    assert Path(public_suffix.DEFAULT_PATH).resolve() == VENDORED.resolve()
