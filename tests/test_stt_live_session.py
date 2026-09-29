"""Behaviour of the live `/ws/transcribe` session that the older WebSocket tests
do not reach: a session longer than the decode window, failures reaching the
client, language handling, idle sockets and how a session holds the model.

Runs the real app under stubs (see test_stt_support.py) with a fake model, so
nothing here needs a GPU or a downloaded model.
"""

import asyncio
import threading
import time

import pytest
from fastapi.testclient import TestClient

from test_stt_support import (
    FakeWhisper, Info, LiveSession, Segment, TimelineWhisper, load_stt_app,
    reset_model, wait_for_preload, wait_until,
)

# Tight limits: WS_WINDOW_S=2 means the decode window slides after only a few
# seconds of audio, which is what the long-session test needs.
ENV = {
    "WS_MIN_NEW_AUDIO_S": "0.1",
    "WS_WINDOW_S": "2.0",
    "WS_MAX_BUFFER_S": "120",
    "WS_MAX_SESSIONS": "2",
}


@pytest.fixture(scope="module")
def stt_app():
    return load_stt_app(ENV, name="stt_app_live_session")


@pytest.fixture(scope="module")
def _client(stt_app):
    with TestClient(stt_app.app) as test_client:
        wait_for_preload(stt_app)
        yield test_client


@pytest.fixture
def client(stt_app, _client):
    """A client with a fresh scripted model; fails the test if it leaks a reference."""
    stt_app._idle_unloader.cancel()
    _client.fake_model = reset_model(stt_app)
    yield _client
    assert wait_until(lambda: stt_app._model_refs == 0 and stt_app._live_sessions == 0), (
        f"a session leaked: refs={stt_app._model_refs} sessions={stt_app._live_sessions}"
    )


def _truth(n_words=40):
    """(start, end, text): a sentence every 8 words, so the window can be cut."""
    out, t = [], 0.0
    for i in range(n_words):
        text = f"wort{i}" + ("." if i % 8 == 7 else "")
        out.append((t, t + 0.3, text))
        t += 0.4 + (0.5 if i % 8 == 7 else 0.0)
    return out


# --- LocalAgreement through the real session ---------------------------------


def test_confirmed_keeps_growing_after_the_decode_window_slides(client, stt_app):
    """The bug this guards: confirmed collapsed to "" once the window rolled.

    Twelve seconds of audio against a 2 s target window (4 s ceiling): the window
    has to slide several times. A model that hears the real timeline is played
    against it, so what reaches the client is exactly what the agreement logic
    made of a sliding window.
    """
    truth = _truth()
    model = TimelineWhisper(truth)
    stt_app.whisper_model = model
    expected = [t for _, _, t in truth]

    with LiveSession(client, {"language": "de"}) as session:
        for i in range(60):                      # 15 s, 0.25 s at a time
            session.send_audio(0.25, start_s=i * 0.25)
            time.sleep(0.02)
        assert wait_until(lambda: len(session.of_type("partial")) >= 5)
        time.sleep(0.3)
        session.send_audio(0.25, start_s=15.0)   # one more tick lets the tail settle
        time.sleep(0.3)
        partials = session.of_type("partial")
        session.stop()
        final = session.wait_type("final")

    confirmed = [p["confirmed"] for p in partials]
    for earlier, later in zip(confirmed, confirmed[1:]):
        assert later.startswith(earlier), "confirmed text was rewritten after being shown"

    words = confirmed[-1].split()
    assert words == expected[:len(words)], "confirmed is not an in-order prefix of what was said"
    spoken_by_12s = [t for start, end, t in truth if end <= 12.0]
    assert len(words) >= len(spoken_by_12s) - 3, (
        f"only {len(words)} of ~{len(spoken_by_12s)} words were confirmed: agreement "
        "stopped working once the window slid"
    )

    interim = [w for w, c in zip(model.windows, model.calls) if c.get("beam_size") == 1]
    assert max(start for start, _ in interim) > 4.0, "the window never slid, so this proved nothing"
    assert max(end - start for start, end in interim) <= 4.6, "the decode window outgrew its ceiling"
    assert final["text"], "the accurate final decode still runs"


def test_agreement_promotes_stable_prefix_to_confirmed(client):
    """Words stable across two consecutive hypotheses become `confirmed`."""
    client.fake_model.script = [
        [Segment("the quick brown")],
        [Segment("the quick green fox")],
    ]
    client.fake_model.default = [Segment("the quick green fox")]

    with LiveSession(client, {"language": "en"}) as session:
        session.send_audio(0.2)
        first = session.wait_type("partial")
        session.send_audio(0.2)
        second = session.wait_for(lambda m: m.get("type") == "partial" and m is not first)

    assert first["confirmed"] == ""          # nothing to agree with yet
    assert second["confirmed"] == "the quick"
    assert second["pending"] == "green fox"


# --- language handling (M6, I1) ----------------------------------------------


def _languages(model):
    return [c.get("language") for c in model.calls]


def test_a_locale_tag_is_reduced_to_the_language_code(client):
    """`de-DE` is what a browser sends; faster-whisper rejects it."""
    client.fake_model.default = [Segment("guten tag")]

    with LiveSession(client, {"language": "de-DE"}) as session:
        session.send_audio(0.3)
        session.wait_type("partial")

    assert _languages(client.fake_model)[0] == "de"


def test_an_unknown_language_is_rejected_out_loud_and_the_old_one_stays(client):
    client.fake_model.default = [Segment("guten tag")]

    with LiveSession(client, {"language": "de"}) as session:
        session.send_json({"language": "klingon"})
        error = session.wait_type("error")
        session.send_audio(0.3)
        session.wait_type("partial")

    assert error["code"] == "invalid_language"
    assert "klingon" in error["message"]
    assert error["error"] == error["message"], "the shipped frontend reads `error`"
    assert set(_languages(client.fake_model)) == {"de"}


def test_the_language_is_locked_after_a_confident_detection(client):
    """Auto-detect used to re-run on every tick over the whole window."""
    client.fake_model.info = Info("de", 0.95)
    client.fake_model.default = [Segment("guten tag zusammen")]

    with LiveSession(client, {"language": "auto"}) as session:
        session.send_audio(0.3)
        session.wait_type("partial")
        for _ in range(4):
            session.send_audio(0.3)
            time.sleep(0.05)
        assert wait_until(lambda: len(client.fake_model.calls) >= 3)
        session.stop()
        session.wait_type("final")

    languages = _languages(client.fake_model)
    assert languages[0] is None
    assert all(lang == "de" for lang in languages[1:]), languages
    assert client.fake_model.calls[-1]["beam_size"] > 1
    assert languages[-1] == "de", "the final decode must reuse the locked language"


def test_an_unsure_detection_does_not_lock(client):
    client.fake_model.info = Info("nn", 0.35)
    client.fake_model.default = [Segment("hmm")]

    with LiveSession(client, {"language": "auto"}) as session:
        for _ in range(4):
            session.send_audio(0.3)
            time.sleep(0.05)
        assert wait_until(lambda: len(client.fake_model.calls) >= 2)

    assert set(_languages(client.fake_model)) == {None}


def test_a_language_the_client_fixed_is_never_replaced_by_detection(client):
    client.fake_model.info = Info("en", 0.99)
    client.fake_model.default = [Segment("guten tag")]

    with LiveSession(client, {"language": "de"}) as session:
        for _ in range(3):
            session.send_audio(0.3)
            time.sleep(0.05)
        assert wait_until(lambda: len(client.fake_model.calls) >= 2)

    assert set(_languages(client.fake_model)) == {"de"}


def test_a_new_language_frame_unlocks_and_applies_from_the_next_decode(client):
    client.fake_model.info = Info("de", 0.95)
    client.fake_model.default = [Segment("hallo")]

    with LiveSession(client, {"language": "auto"}) as session:
        session.send_audio(0.3)
        session.wait_type("partial")
        session.send_json({"language": "en"})
        time.sleep(0.05)
        calls_before = len(client.fake_model.calls)
        session.send_audio(0.3)
        assert wait_until(lambda: len(client.fake_model.calls) > calls_before)

    assert _languages(client.fake_model)[-1] == "en"


# --- failures reach the client (M6) ------------------------------------------


def test_repeated_interim_failures_are_reported_and_the_session_survives(client, stt_app):
    client.fake_model.fail_with = RuntimeError("CUDA error: an illegal memory access")

    with LiveSession(client, {"language": "de"}) as session:
        session.send_audio(0.3)
        for _ in range(stt_app.WS_MAX_INTERIM_ERRORS + 2):
            time.sleep(0.05)
            session.send_audio(0.3)
        error = session.wait_type("error")
        assert error["code"] == "decode_failed"
        assert "illegal memory access" in error["message"]

        # Still open, and it recovers as soon as decoding does.
        client.fake_model.fail_with = None
        client.fake_model.default = [Segment("wieder da")]
        for _ in range(6):
            session.send_audio(0.3)
            time.sleep(0.05)
        partial = session.wait_type("partial")
        assert "wieder" in (partial["confirmed"] + " " + partial["pending"])
        assert session.closed is None


def test_one_error_per_failure_streak_not_one_per_tick(client, stt_app):
    client.fake_model.fail_with = RuntimeError("boom")

    with LiveSession(client, {"language": "de"}) as session:
        for _ in range(stt_app.WS_MAX_INTERIM_ERRORS * 3):
            session.send_audio(0.3)
            time.sleep(0.03)
        assert wait_until(lambda: len(client.fake_model.calls) >= stt_app.WS_MAX_INTERIM_ERRORS * 2)
        time.sleep(0.2)
        assert len(session.of_type("error")) == 1


def test_a_model_that_cannot_load_is_reported_at_the_handshake(client, stt_app, monkeypatch):
    class Broken:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("weights are corrupt")

    monkeypatch.setattr(stt_app, "WhisperModel", Broken)
    stt_app.whisper_model = None
    stt_app.model_loaded = False

    with LiveSession(client) as session:
        error = session.wait_type("error")
        closed = session.wait_closed()

    assert error["code"] == "model_unavailable"
    assert "weights are corrupt" in error["message"]
    assert getattr(closed, "code", None) == 1011
    stt_app.startup_error = None


def test_a_503_from_the_model_mid_session_reaches_the_client(client, stt_app):
    from fastapi import HTTPException

    client.fake_model.fail_with = HTTPException(status_code=503, detail="Model not available: gone")

    with LiveSession(client, {"language": "de"}) as session:
        for _ in range(stt_app.WS_MAX_INTERIM_ERRORS + 2):
            session.send_audio(0.3)
            time.sleep(0.05)
        error = session.wait_type("error")

    assert "Model not available: gone" in error["message"]


# --- idle sockets ------------------------------------------------------------


def test_a_silent_socket_is_closed_with_a_reason_and_frees_its_slot(client, stt_app, monkeypatch):
    monkeypatch.setattr(stt_app, "WS_IDLE_TIMEOUT_S", 0.3)

    with LiveSession(client) as session:
        error = session.wait_type("error", timeout=3)
        closed = session.wait_closed(timeout=3)

    assert error["code"] == "idle_timeout"
    assert getattr(closed, "code", None) == 4408
    assert "idle" in (getattr(closed, "reason", "") or "")
    assert wait_until(lambda: stt_app._live_sessions == 0)


def test_traffic_keeps_a_session_alive(client, stt_app, monkeypatch):
    monkeypatch.setattr(stt_app, "WS_IDLE_TIMEOUT_S", 0.5)
    client.fake_model.default = [Segment("noch da")]

    with LiveSession(client, {"language": "de"}) as session:
        for _ in range(12):                  # 1.8 s in total, 3.6x the timeout
            session.send_audio(0.1)
            time.sleep(0.15)
        assert session.closed is None
        assert not session.of_type("error")


def test_idle_timeout_can_be_switched_off(client, stt_app, monkeypatch):
    monkeypatch.setattr(stt_app, "WS_IDLE_TIMEOUT_S", 0)

    with LiveSession(client, {"language": "de"}) as session:
        time.sleep(0.4)
        assert session.closed is None


# --- how a session holds the model (M4) --------------------------------------


def test_a_session_holds_the_model_and_lets_go_when_it_ends(client, stt_app):
    with LiveSession(client, {"language": "de"}) as session:
        assert wait_until(lambda: stt_app._model_refs >= 1), "the session does not pin the model"
        assert client.get("/health").json()["model_refs"] >= 1
        session.stop()
        session.wait_type("final")
    assert wait_until(lambda: stt_app._model_refs == 0)


def test_a_refused_session_does_not_load_the_model(client, stt_app, monkeypatch):
    constructed = []

    class Counting(FakeWhisper):
        def __init__(self, *args, **kwargs):
            constructed.append(1)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(stt_app, "WhisperModel", Counting)
    stt_app.whisper_model = None
    stt_app.model_loaded = False
    stt_app._live_sessions = stt_app.WS_MAX_SESSIONS   # the limit is reached

    try:
        with LiveSession(client) as session:
            error = session.wait_type("error")
    finally:
        stt_app._live_sessions = 0

    assert error["code"] == "too_many_sessions"
    assert "Too many live sessions" in error["error"]
    assert constructed == [], "a refused session loaded the model"


def test_a_cold_model_load_does_not_freeze_the_event_loop(client, stt_app, monkeypatch):
    """A handshake that finds the model unloaded loads it under the reference lock,
    for seconds to minutes. On the loop that froze /health and every other session."""
    entered, gate = threading.Event(), threading.Event()

    class Slow(FakeWhisper):
        def __init__(self, *args, **kwargs):
            entered.set()
            gate.wait(15)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(stt_app, "WhisperModel", Slow)
    stt_app.whisper_model = None
    stt_app.model_loaded = False

    # Opened from a thread: against a blocked loop even the connect call hangs,
    # and this test has to be the one that measures that, not suffer it.
    opened = {}
    opener = threading.Thread(
        target=lambda: opened.setdefault("session", LiveSession(client, {"language": "de"})),
        daemon=True,
    )
    opener.start()
    try:
        assert entered.wait(5), "the handshake never started a load"

        answers = {}

        def probe():
            started = time.monotonic()
            answers["health"] = client.get("/health")
            answers["ready"] = client.get("/ready")
            answers["elapsed"] = time.monotonic() - started

        prober = threading.Thread(target=probe, daemon=True)
        prober.start()
        prober.join(3)
        assert not prober.is_alive() and answers["elapsed"] < 2.0, (
            "/health did not answer while a model load was in progress: the event loop is blocked"
        )
        assert answers["health"].status_code == 200
        assert answers["ready"].status_code == 503
        assert answers["ready"].json()["reason"] == "loading"
    finally:
        gate.set()
        opener.join(10)
        if "session" in opened:
            opened["session"].close()


def test_a_cancelled_acquire_does_not_leak_its_reference(stt_app):
    """A worker thread cannot be interrupted: when the awaiting task is cancelled
    mid-load the thread still takes its reference, so it has to be handed back."""
    stt_app._idle_unloader.cancel()
    gate = threading.Event()
    real_acquire = stt_app.acquire_model

    def slow_acquire():
        gate.wait(5)
        return real_acquire()

    stt_app.acquire_model = slow_acquire
    reset_model(stt_app)

    async def scenario():
        task = asyncio.create_task(stt_app._hold_model_async())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        gate.set()
        for _ in range(200):
            if stt_app._model_refs == 0:
                return
            await asyncio.sleep(0.01)
        raise AssertionError(f"the cancelled acquire leaked a reference: {stt_app._model_refs}")

    try:
        asyncio.run(scenario())
    finally:
        stt_app.acquire_model = real_acquire
        gate.set()
        wait_until(lambda: stt_app._model_refs == 0)


# --- a model that unloads the moment it is idle ------------------------------


def test_ttl_zero_loads_the_model_once_per_session_not_twice():
    """With STT_MODEL_TTL=0 a probe's release unloads the model immediately, so a
    handshake that probed and then decoded loaded it twice."""
    constructed = []

    class Counting(FakeWhisper):
        def __init__(self, *args, **kwargs):
            constructed.append(1)
            super().__init__(*args, **kwargs)
            self.default = [Segment("guten tag")]

    app = load_stt_app({**ENV, "STT_MODEL_TTL": "0"}, name="stt_app_live_ttl0", model_cls=Counting)
    assert app.MODEL_TTL == 0
    with TestClient(app.app) as test_client:
        wait_for_preload(app)
        assert constructed == [], "TTL=0 must not preload"
        with LiveSession(test_client, {"language": "de"}) as session:
            session.send_audio(0.3)
            session.wait_type("partial")
            session.stop()
            session.wait_type("final")
        assert wait_until(lambda: app._model_refs == 0)
        assert wait_until(lambda: app.whisper_model is None), "the idle model was not released"

    assert constructed == [1], f"the model was loaded {len(constructed)} times for one session"


def test_a_configured_default_language_applies_until_the_client_says_otherwise():
    app = load_stt_app({**ENV, "STT_DEFAULT_LANGUAGE": "de-DE"}, name="stt_app_live_default_lang")
    assert app.STT_DEFAULT_LANGUAGE == "de"
    with TestClient(app.app) as test_client:
        wait_for_preload(app)

        model = reset_model(app)
        model.default = [Segment("hallo")]
        with LiveSession(test_client) as session:                 # names no language
            session.send_audio(0.3)
            session.wait_type("partial")
        assert {c.get("language") for c in model.calls} == {"de"}

        model = reset_model(app)
        model.default = [Segment("hello")]
        with LiveSession(test_client, {"language": "auto"}) as session:   # explicit auto wins
            session.send_audio(0.3)
            session.wait_type("partial")
        assert model.calls[0].get("language") is None

        model = reset_model(app)
        model.default = [Segment("hello")]
        with LiveSession(test_client, {"language": "en-GB"}) as session:
            session.send_audio(0.3)
            session.wait_type("partial")
        assert {c.get("language") for c in model.calls} == {"en"}
