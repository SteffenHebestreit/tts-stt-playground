"""Offline regression tests for the live-transcription WebSocket state machine.

These cover the realtime behaviours that had no test coverage at all and that
are invisible to the existing integration suite (which needs live services):

- ingest and decode are decoupled, and lag does not grow without bound
- the session buffer rolls instead of freezing at the cap
- silence does not blank an already-rendered transcript
- hallucinated segments are filtered before they can be promoted to "confirmed"
- the final decode is the accurate one, not the greedy interim one

The Whisper model is stubbed, so nothing here needs a GPU or a downloaded model.
torch/faster-whisper are stubbed too; numpy is genuinely required.
"""

import os
import sys
import time
import types
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

pytest.importorskip("numpy", reason="stt-service app.py requires numpy")

import numpy as np  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

STT_DIR = Path(__file__).resolve().parents[1] / "stt-service"

# Tight limits so tests exercise the boundaries without pushing real-time audio.
TEST_ENV = {
    "WS_MIN_NEW_AUDIO_S": "0.1",
    "WS_WINDOW_S": "2.0",
    "WS_MAX_BUFFER_S": "1.0",
    "WS_MAX_SESSIONS": "2",
    "USE_CUDA": "false",
    "WHISPER_MODEL_SIZE": "tiny",
}


class _Segment:
    """Minimal stand-in for a faster-whisper segment."""

    def __init__(self, text, no_speech_prob=0.0, avg_logprob=-0.2):
        self.text = text
        self.no_speech_prob = no_speech_prob
        self.avg_logprob = avg_logprob
        self.start = 0.0
        self.end = 1.0


class _Info:
    language = "en"
    language_probability = 0.99
    duration = 1.0


class FakeWhisper:
    """Returns scripted hypotheses and records the kwargs it was called with.

    `default` is returned once `script` is exhausted. Decodes are single-flight
    and skip-ahead by design, so a test can never assume one decode per audio
    frame — `default` lets a test stay deterministic under that coalescing.
    """

    def __init__(self, *args, **kwargs):
        self.script = []
        self.default = []
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        segments = self.script.pop(0) if self.script else self.default
        return iter(segments), _Info()


def _install_stubs():
    """Stub torch and faster_whisper so app.py imports without a GPU stack."""
    if "torch" not in sys.modules:
        torch = types.ModuleType("torch")
        torch.__version__ = "0.0.0-stub"
        cuda = types.SimpleNamespace(
            is_available=lambda: False,
            device_count=lambda: 0,
            get_device_name=lambda i: "stub",
        )
        torch.cuda = cuda
        torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
        sys.modules["torch"] = torch

    if "faster_whisper" not in sys.modules:
        fw = types.ModuleType("faster_whisper")
        fw.WhisperModel = FakeWhisper
        sys.modules["faster_whisper"] = fw


def _load_app_module():
    """Import stt-service/app.py under a stubbed environment."""
    _install_stubs()
    for key, value in TEST_ENV.items():
        os.environ[key] = value

    # app.py does `from json_utils import ...`, which lives beside it.
    sys.path.insert(0, str(STT_DIR))
    try:
        spec = spec_from_file_location("stt_app_under_test", STT_DIR / "app.py")
        module = module_from_spec(spec)
        assert spec.loader is not None
        sys.modules["stt_app_under_test"] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(STT_DIR))


@pytest.fixture(scope="module")
def stt_app():
    return _load_app_module()


@pytest.fixture(scope="module")
def _test_client(stt_app):
    """One TestClient for the whole module.

    Deliberately module-scoped: the service's lifespan shutdown calls
    `executor.shutdown()`, so a per-test client would leave every later test
    unable to schedule a decode (and hanging on the partial that never arrives).
    """
    with TestClient(stt_app.app) as test_client:
        yield test_client


@pytest.fixture
def client(stt_app, _test_client):
    """Reset the scripted fake model and session state before each test."""
    model = FakeWhisper()
    stt_app.whisper_model = model
    stt_app.model_loaded = True
    stt_app._live_sessions = 0
    _test_client.fake_model = model
    return _test_client


def pcm(seconds: float, rate: int = 16000) -> bytes:
    """`seconds` of quiet noise as PCM16 little-endian mono."""
    rng = np.random.default_rng(0)
    samples = (rng.standard_normal(int(seconds * rate)) * 1000).astype(np.int16)
    return samples.tobytes()


def _drain_until(ws, msg_type, limit=10):
    """Read messages until one of `msg_type` arrives; fail if it never does."""
    for _ in range(limit):
        message = ws.receive_json()
        if message.get("type") == msg_type:
            return message
    pytest.fail(f"no {msg_type!r} message within {limit} messages")


def test_partial_reports_latency_metrics(client):
    """Partials must carry decode timings — without them nothing else is tunable."""
    client.fake_model.script = [[_Segment("hello world")]]

    with client.websocket_connect("/ws/transcribe") as ws:
        ws.send_json({"language": "en"})
        ws.send_bytes(pcm(0.2))
        message = _drain_until(ws, "partial")

    assert message["pending"] or message["confirmed"]
    for key in ("decode_ms", "lag_ms", "pending_seconds", "buffered_seconds"):
        assert key in message, f"partial is missing {key}"
        assert isinstance(message[key], (int, float))


def test_agreement_promotes_stable_prefix_to_confirmed(client):
    """Words stable across two consecutive hypotheses become `confirmed`."""
    client.fake_model.script = [
        [_Segment("the quick brown")],
        [_Segment("the quick green fox")],
    ]

    with client.websocket_connect("/ws/transcribe") as ws:
        ws.send_json({"language": "en"})
        ws.send_bytes(pcm(0.2))
        first = _drain_until(ws, "partial")
        ws.send_bytes(pcm(0.2))
        second = _drain_until(ws, "partial")

    # Nothing is confirmed on the first hypothesis — there is nothing to agree with.
    assert first["confirmed"] == ""
    # "the quick" is common to both; "green fox" is still pending.
    assert second["confirmed"] == "the quick"
    assert second["pending"] == "green fox"


def test_silence_does_not_blank_the_transcript(client):
    """An empty hypothesis must not clear a transcript the user is reading."""
    client.fake_model.script = [
        [_Segment("some real speech")],
        [],  # VAD filtered everything: a pause
    ]
    # Every decode after the pause resumes speech. Decodes coalesce, so the test
    # must not depend on exactly how many fire.
    client.fake_model.default = [_Segment("some real speech continues")]

    with client.websocket_connect("/ws/transcribe") as ws:
        ws.send_json({"language": "en"})
        ws.send_bytes(pcm(0.2))
        first = _drain_until(ws, "partial")

        # Drive enough decodes that the empty one is definitely consumed and at
        # least one more has run.
        for _ in range(8):
            ws.send_bytes(pcm(0.2))
            time.sleep(0.05)
        second = _drain_until(ws, "partial")

    assert first["pending"] == "some real speech"
    # A blank partial would have been sent had the empty decode not been skipped,
    # and this assertion would read "" instead.
    assert "some real speech" in (second["confirmed"] + " " + second["pending"])
    # The empty hypothesis was decoded but produced no message.
    assert len(client.fake_model.calls) >= 3


def test_hallucinated_segments_are_filtered(client):
    """High no-speech / low-logprob segments never reach the client.

    They repeat across windows, and the agreement check would otherwise promote
    them to `confirmed` precisely *because* they repeat.
    """
    client.fake_model.script = [
        [
            _Segment("Thank you for watching.", no_speech_prob=0.95),
            _Segment("real content", no_speech_prob=0.01),
            _Segment("mumble", avg_logprob=-3.0),
        ],
    ]

    with client.websocket_connect("/ws/transcribe") as ws:
        ws.send_json({"language": "en"})
        ws.send_bytes(pcm(0.2))
        message = _drain_until(ws, "partial")

    text = f"{message['confirmed']} {message['pending']}"
    assert "real content" in text
    assert "Thank you for watching" not in text
    assert "mumble" not in text


def test_buffer_rolls_instead_of_freezing(client, stt_app):
    """Past the cap the session must keep working, not go silently dead.

    The previous implementation truncated *new* audio, which froze the sample
    counter so the decode gate could never fire again — the session produced no
    further output while the client kept uploading.
    """
    assert stt_app.WS_MAX_BUFFER_S == 1.0, "test env not applied"
    client.fake_model.script = [[_Segment(f"chunk {i}")] for i in range(10)]

    with client.websocket_connect("/ws/transcribe") as ws:
        ws.send_json({"language": "en"})
        ws.send_bytes(pcm(0.6))
        _drain_until(ws, "partial")

        # Push well past WS_MAX_BUFFER_S.
        ws.send_bytes(pcm(0.6))
        ws.send_bytes(pcm(0.6))

        saw_warning = False
        saw_partial_after_cap = False
        for _ in range(8):
            message = ws.receive_json()
            if message.get("type") == "warning" and message.get("code") == "buffer_rolled":
                saw_warning = True
            elif message.get("type") == "partial":
                saw_partial_after_cap = True
            if saw_warning and saw_partial_after_cap:
                break
            ws.send_bytes(pcm(0.3))

    assert saw_warning, "client was never told the buffer rolled"
    assert saw_partial_after_cap, "session stopped producing partials past the cap"


def test_final_decode_uses_accurate_settings(client):
    """The final transcript must not reuse the greedy interim settings."""
    client.fake_model.script = [
        [_Segment("interim guess")],
        [_Segment("accurate final transcript")],
    ]

    with client.websocket_connect("/ws/transcribe") as ws:
        ws.send_json({"language": "en"})
        ws.send_bytes(pcm(0.2))
        _drain_until(ws, "partial")
        ws.send_json({"event": "stop"})
        final = _drain_until(ws, "final")

    assert final["text"] == "accurate final transcript"
    assert final["language"] == "en"

    interim_kwargs, final_kwargs = client.fake_model.calls[0], client.fake_model.calls[-1]
    assert interim_kwargs["beam_size"] == 1, "interim decode should stay greedy"
    assert final_kwargs["beam_size"] > 1, "final decode should use beam search"
    assert final_kwargs["condition_on_previous_text"] is True


def test_empty_session_still_returns_a_final(client):
    """Stopping without sending audio must close cleanly, not hang."""
    with client.websocket_connect("/ws/transcribe") as ws:
        ws.send_json({"event": "stop"})
        final = _drain_until(ws, "final")

    assert final["text"] == ""
    assert final["duration"] == 0.0


def test_session_limit_is_enforced(client, stt_app):
    """Admission control refuses excess live sessions instead of thrashing."""
    assert stt_app.WS_MAX_SESSIONS == 2, "test env not applied"

    with client.websocket_connect("/ws/transcribe") as first:
        first.send_json({"language": "en"})
        with client.websocket_connect("/ws/transcribe") as second:
            second.send_json({"language": "en"})
            with client.websocket_connect("/ws/transcribe") as third:
                message = third.receive_json()

    assert message["type"] == "error"
    assert "Too many live sessions" in message["error"]


def test_health_reports_503_only_when_loading_actually_failed(client, stt_app):
    """A service whose model failed to load must not advertise itself as healthy."""
    original_loaded, original_error = stt_app.model_loaded, stt_app.startup_error
    try:
        stt_app.model_loaded = False
        stt_app.startup_error = "simulated load failure"
        response = client.get("/health")
        assert response.status_code == 503
        body = response.json()
        assert body["model_loaded"] is False
        assert body["can_load"] is False
        assert body["startup_error"] == "simulated load failure"
    finally:
        stt_app.model_loaded, stt_app.startup_error = original_loaded, original_error

    assert client.get("/health").status_code == 200


def test_health_stays_200_when_the_model_is_merely_not_resident(client, stt_app):
    """An unloaded-but-loadable model is NOT an outage.

    Once idle-TTL unloading exists, a quiet service legitimately holds no model.
    Docker's healthcheck uses `curl -f`, which fails on 503, and the gateway
    derives health from the status code — so returning 503 here would mark an
    idle container unhealthy and show it as down in the UI.
    """
    original_model, original_loaded = stt_app.whisper_model, stt_app.model_loaded
    try:
        stt_app.whisper_model = None
        stt_app.model_loaded = False
        stt_app.startup_error = None

        response = client.get("/health")
        assert response.status_code == 200, "an idle service must not report as unhealthy"
        body = response.json()
        assert body["model_resident"] is False
        assert body["can_load"] is True
        assert body["status"] == "ok"
    finally:
        stt_app.whisper_model, stt_app.model_loaded = original_model, original_loaded


def test_compute_type_defaults_to_a_memory_saving_type(stt_app, monkeypatch):
    """Default must not be plain float16 — int8 is ~35% less VRAM at equal speed,
    which decides whether an 8-12 GB card fits the model at all."""
    monkeypatch.delenv("WHISPER_COMPUTE_TYPE", raising=False)
    assert stt_app._select_cuda_compute_type().startswith("int8")


def test_compute_type_env_override_is_honoured(stt_app, monkeypatch):
    monkeypatch.setenv("WHISPER_COMPUTE_TYPE", "float32")
    assert stt_app._select_cuda_compute_type() == "float32"


def test_compute_type_auto_falls_through_to_selection(stt_app, monkeypatch):
    monkeypatch.setenv("WHISPER_COMPUTE_TYPE", "auto")
    assert stt_app._select_cuda_compute_type().startswith("int8")


def test_oom_ladder_never_escalates_model_size(stt_app, monkeypatch):
    """Every fallback rung must ask for LESS than the one before it.

    The original ladder fell from large-v3-turbo (~1.6 GB) to large-v3
    (~3.1 GB) — responding to a failure by demanding twice the memory, which
    can never help on a constrained device.
    """
    attempts = []

    class _AlwaysFails:
        def __init__(self, model_size, device=None, compute_type=None, **kwargs):
            attempts.append((model_size, device, compute_type))
            raise RuntimeError("simulated CUDA OOM")

    saved = (stt_app.WhisperModel, stt_app.whisper_model, stt_app.model_loaded,
             stt_app.startup_error, stt_app.device, stt_app.compute_type)
    try:
        monkeypatch.setenv("WHISPER_MODEL_SIZE", "large-v3-turbo")
        stt_app.WhisperModel = _AlwaysFails
        stt_app.device, stt_app.compute_type = "cuda", "int8_float16"
        stt_app.load_model()
    finally:
        (stt_app.WhisperModel, stt_app.whisper_model, stt_app.model_loaded,
         stt_app.startup_error, stt_app.device, stt_app.compute_type) = saved

    assert attempts, "load_model made no attempts"
    # Once the ladder leaves the GPU it must never go back.
    devices = [d for _, d, _ in attempts]
    assert devices == sorted(devices, key=lambda d: 0 if d == "cuda" else 1), \
        f"ladder returned to the GPU after falling back to CPU: {devices}"

    # NO rung may ask for a bigger model than the one that was requested.
    # Deliberately checks every rung including index 1 — an earlier version of
    # this test exempted it, and that is exactly where the escalation hid.
    requested_rank = stt_app._model_rank(attempts[0][0])
    for i, (model_size, dev, _) in enumerate(attempts):
        assert stt_app._model_rank(model_size) <= requested_rank, (
            f"rung {i} ({model_size} on {dev}) is LARGER than the requested "
            f"{attempts[0][0]}; a fallback must never escalate: {attempts}"
        )

    # And the English-only distil model must never be a rung — it would
    # silently drop German on exactly the devices this ladder targets.
    assert not any("distil" in m for m, _, _ in attempts), attempts


def test_unknown_model_name_still_falls_back_to_a_known_one(stt_app, monkeypatch):
    """An unrecognised alias must not brick startup.

    This is the one case where a size *increase* is acceptable: the requested
    model does not exist, so there is nothing smaller to step down to.
    """
    attempts = []

    class _RecordsThenSucceeds:
        def __init__(self, model_size, device=None, compute_type=None, **kwargs):
            attempts.append(model_size)
            if model_size not in stt_app.KNOWN_MODEL_SIZES:
                raise RuntimeError("invalid model name")

    saved = (stt_app.WhisperModel, stt_app.whisper_model, stt_app.model_loaded,
             stt_app.startup_error, stt_app.device, stt_app.compute_type)
    try:
        monkeypatch.setenv("WHISPER_MODEL_SIZE", "not-a-real-model")
        stt_app.WhisperModel = _RecordsThenSucceeds
        stt_app.device, stt_app.compute_type = "cuda", "int8_float16"
        stt_app.load_model()
        loaded = stt_app.model_size_loaded
    finally:
        (stt_app.WhisperModel, stt_app.whisper_model, stt_app.model_loaded,
         stt_app.startup_error, stt_app.device, stt_app.compute_type) = saved

    assert attempts[0] == "not-a-real-model"
    assert stt_app.FALLBACK_MODEL_SIZE in attempts
    assert loaded == stt_app.FALLBACK_MODEL_SIZE


def test_translate_is_rejected_on_turbo_models(client, stt_app):
    """Turbo models cannot translate; the service must say so instead of
    silently returning source-language text."""
    original = stt_app.model_size_loaded
    try:
        stt_app.model_size_loaded = "large-v3-turbo"
        with pytest.raises(Exception) as excinfo:
            stt_app._reject_unsupported_translate("translate")
        assert excinfo.value.status_code == 400
        assert "does not support translation" in excinfo.value.detail

        # transcribe is unaffected.
        stt_app._reject_unsupported_translate("transcribe")

        stt_app.model_size_loaded = "large-v3"
        stt_app._reject_unsupported_translate("translate")
    finally:
        stt_app.model_size_loaded = original


def test_control_frame_without_language_does_not_reset_it(client):
    """A control frame that omits `language` must leave the session language alone.

    The language line used to run on every non-stop text frame, so any control
    message that did not repeat the language — a keepalive, a future event type —
    silently reset a German session to auto-detect and told no one. The browser
    only ever sends {language} then {event:"stop"}, so it never tripped this;
    an API client sending anything else did, and this is used mostly as an API.

    Auto-detect on short or noisy German is exactly where Whisper guesses
    English, so the failure is silent and German-shaped.
    """
    client.fake_model.default = [_Segment("guten tag")]

    with client.websocket_connect("/ws/transcribe") as ws:
        ws.send_json({"language": "de"})
        ws.send_bytes(pcm(0.2))
        _drain_until(ws, "partial")

        # Any other control frame — no `language` key anywhere in it.
        ws.send_json({"event": "ping"})
        ws.send_bytes(pcm(0.4))
        _drain_until(ws, "partial")

    languages = [call.get("language") for call in client.fake_model.calls]
    assert languages, "no decode ran"
    assert all(lang == "de" for lang in languages), (
        f"the session language was reset by a control frame: {languages}"
    )


def test_control_frame_can_still_change_the_language(client):
    """The reset guard must not make `language` immutable mid-session."""
    client.fake_model.default = [_Segment("hello")]

    with client.websocket_connect("/ws/transcribe") as ws:
        ws.send_json({"language": "de"})
        ws.send_bytes(pcm(0.2))
        _drain_until(ws, "partial")
        first = [c.get("language") for c in client.fake_model.calls]

        ws.send_json({"language": "en"})
        ws.send_bytes(pcm(0.4))
        _drain_until(ws, "partial")

    languages = [c.get("language") for c in client.fake_model.calls]
    assert first and first[0] == "de"
    assert languages[-1] == "en", f"explicit language change was ignored: {languages}"


def test_explicit_auto_still_means_auto_detect(client):
    """`{"language": "auto"}` must map to None, not to the literal string."""
    client.fake_model.default = [_Segment("hello")]

    with client.websocket_connect("/ws/transcribe") as ws:
        ws.send_json({"language": "auto"})
        ws.send_bytes(pcm(0.2))
        _drain_until(ws, "partial")

    languages = [c.get("language") for c in client.fake_model.calls]
    assert languages and all(lang is None for lang in languages), (
        f"faster-whisper rejects the literal 'auto'; got {languages}"
    )
