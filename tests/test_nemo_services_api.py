"""Response contracts of the two NeMo ASR services: formats, batches, languages, errors.

Findings covered:

* ``response_format=text`` on the OpenAI route returned a JSON-quoted string;
* /transcribe-batch answered ``null`` entries when NeMo returned fewer hypotheses
  than files, and one unreadable file sank the whole batch;
* canary silently decoded any unsupported language (Italian, say) as German, and
  its ``except TypeError`` re-ran a whole second decode on any TypeError;
* CUDA out-of-memory came back as a generic 500.
"""

from __future__ import annotations

import logging
import re
import wave

import pytest

from test_nemo_services_harness import (
    FakeCanary, FakeCanaryV1, FakeCanaryWithoutTimestamps, FakeParakeet, OutOfMemoryError,
    hypothesis, install_model, load_app, make_client, run, wav_bytes,
)

SERVICES = ["parakeet", "canary"]
MODELS = {"parakeet": FakeParakeet, "canary": FakeCanary}


@pytest.fixture(params=SERVICES)
def service(request):
    return request.param


def build(service, monkeypatch, *, model=None, **env):
    module = load_app(service, **env)
    monkeypatch.setattr(module, "_FFMPEG", None)  # every clip here is a native 16 kHz mono WAV
    model = model if model is not None else MODELS[service]()
    install_model(module, model)
    return module, model


async def post(client, data, *, name="clip.wav", path="/transcribe", field="audio", **form):
    return await client.post(path, files={field: (name, data, "audio/wav")}, data=form)


async def post_batch(client, clips, **form):
    files = [("audios", (name, data, "audio/wav")) for name, data in clips]
    return await client.post("/transcribe-batch", files=files, data=form)


# --- N2: response_format=text --------------------------------------------------------

def test_text_format_is_the_raw_transcript_not_a_json_string(service, monkeypatch):
    module, model = build(service, monkeypatch)
    model.reply = lambda paths, kwargs: [hypothesis("hallo welt")]

    async def scenario():
        async with make_client(module) as client:
            return await post(client, wav_bytes(1.0), path="/v1/audio/transcriptions", field="file",
                              response_format="text")

    response = run(scenario())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert response.text == "hallo welt", "the body must be the transcript itself, without JSON quotes"


def test_the_json_format_still_carries_text_and_segments(service, monkeypatch):
    module, model = build(service, monkeypatch)
    model.reply = lambda paths, kwargs: [hypothesis("hallo welt")]

    async def scenario():
        async with make_client(module) as client:
            return await post(client, wav_bytes(1.0), path="/v1/audio/transcriptions", field="file")

    body = run(scenario()).json()

    assert body["text"] == "hallo welt"
    assert body["segments"] == [{"start": 0.0, "end": 1.0, "text": "hallo welt"}]


def test_transcribe_keeps_the_stt_form_v1_shape(service, monkeypatch):
    module, model = build(service, monkeypatch)
    model.reply = lambda paths, kwargs: [hypothesis("hallo welt")]

    async def scenario():
        async with make_client(module) as client:
            return await post(client, wav_bytes(2.0))

    body = run(scenario()).json()

    assert set(body) >= {"text", "segments", "language", "language_probability", "duration",
                         "processing_time", "task", "model"}
    assert body["duration"] == pytest.approx(2.0)
    assert body["segments"][0]["text"] == "hallo welt"


# --- errors --------------------------------------------------------------------------

def test_cuda_out_of_memory_is_a_503_that_says_so_and_frees_the_cache(service, monkeypatch):
    module, model = build(service, monkeypatch, cuda=True)

    def explode(paths, kwargs):
        raise OutOfMemoryError("CUDA out of memory. Tried to allocate 9.00 GiB")

    model.reply = explode

    async def scenario():
        async with make_client(module) as client:
            return await post(client, wav_bytes(1.0))

    response = run(scenario())

    assert response.status_code == 503
    assert "out of memory" in response.json()["detail"].lower()
    assert ("empty_cache",) in module.torch.calls


def test_other_model_errors_are_a_500_that_names_a_request_id_and_not_the_message(service, monkeypatch, caplog):
    module, model = build(service, monkeypatch)

    def explode(paths, kwargs):
        raise RuntimeError("decoder exploded reading /tmp/tmpab12cd.wav")

    model.reply = explode

    async def scenario():
        async with make_client(module) as client:
            return await post(client, wav_bytes(1.0))

    with caplog.at_level(logging.ERROR):
        response = run(scenario())

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert "decoder exploded" not in detail and "/tmp/" not in detail, "the exception text is for the log"
    request_id = re.search(r"Request id: ([0-9a-f]{12})\.", detail).group(1)
    logged = [r for r in caplog.records if request_id in r.getMessage()]
    assert logged and "decoder exploded reading /tmp/tmpab12cd.wav" in caplog.text, (
        "the operator must be able to find the exception under the id the client was given"
    )


# --- N4 / batch ----------------------------------------------------------------------

def _batch_of(n, seconds=1.0):
    return [(f"clip{i}.wav", wav_bytes(seconds)) for i in range(n)]


def test_batch_reports_every_file_and_keeps_their_order(service, monkeypatch):
    module, model = build(service, monkeypatch)

    async def scenario():
        async with make_client(module) as client:
            return await post_batch(client, _batch_of(3))

    body = run(scenario()).json()

    assert body["batch"] is True and body["file_count"] == 3
    assert [r["filename"] for r in body["results"]] == ["clip0.wav", "clip1.wav", "clip2.wav"]
    assert all("text" in r and "duration" in r for r in body["results"])
    assert "batch_processing_time" in body or service == "canary"


def test_a_file_over_the_duration_limit_fails_alone_in_a_batch(service, monkeypatch):
    module, model = build(service, monkeypatch, NEMO_MAX_AUDIO_S="5")

    async def scenario():
        async with make_client(module) as client:
            return await post_batch(client, [("ok.wav", wav_bytes(2.0)), ("long.wav", wav_bytes(9.0)),
                                             ("ok2.wav", wav_bytes(1.0))])

    body = run(scenario()).json()

    ok, long_, ok2 = body["results"]
    assert "text" in ok and "text" in ok2
    assert long_["filename"] == "long.wav" and long_["status"] == 413
    assert "NEMO_MAX_AUDIO_S" in long_["error"]


def test_a_file_over_the_size_limit_fails_alone_in_a_batch(service, monkeypatch):
    module, model = build(service, monkeypatch, MAX_UPLOAD_MB="0.05")

    async def scenario():
        async with make_client(module) as client:
            return await post_batch(client, [("ok.wav", wav_bytes(1.0)), ("big.wav", wav_bytes(6.0))])

    ok, big = run(scenario()).json()["results"]

    assert "text" in ok
    assert big["status"] == 413 and "MAX_UPLOAD_MB" in big["error"]


def test_fewer_hypotheses_than_files_fail_the_missing_ones_explicitly():
    """NeMo returned two results for three files: the third used to be `null`."""
    module = load_app("parakeet")
    module._FFMPEG = None
    model = FakeParakeet()
    model.reply = lambda paths, kwargs: [hypothesis(f"text{i}") for i in range(len(paths) - 1)]
    install_model(module, model)

    async def scenario():
        async with make_client(module) as client:
            return await post_batch(client, _batch_of(3))

    results = run(scenario()).json()["results"]

    assert all(isinstance(entry, dict) for entry in results), results
    assert [entry.get("text") for entry in results[:2]] == ["text0", "text1"]
    assert results[2]["filename"] == "clip2.wav"
    assert results[2]["status"] == 500 and "no transcription" in results[2]["error"]


def test_a_batch_that_fails_as_a_whole_is_retried_file_by_file():
    """One unreadable file must not sink the others."""
    module = load_app("parakeet")
    module._FFMPEG = None
    model = FakeParakeet()

    def is_the_bad_clip(path):  # the bad clip is the 3 s one
        with wave.open(path, "rb") as fh:
            return fh.getnframes() == 3 * fh.getframerate()

    def reply(paths, kwargs):
        if len(paths) > 1 or any(is_the_bad_clip(p) for p in paths):
            raise RuntimeError("cannot decode")
        return [hypothesis("fine")]

    model.reply = reply
    install_model(module, model)

    async def scenario():
        async with make_client(module) as client:
            return await post_batch(client, [("a.wav", wav_bytes(1.0)), ("bad.wav", wav_bytes(3.0)),
                                             ("b.wav", wav_bytes(1.0))])

    results = run(scenario()).json()["results"]

    assert results[0]["text"] == "fine" and results[2]["text"] == "fine"
    assert results[1]["filename"] == "bad.wav" and results[1]["status"] == 500
    assert "cannot decode" not in results[1]["error"] and "Request id:" in results[1]["error"]
    # one batched attempt over all three, then one call per file
    assert [len(call["paths"]) for call in model.calls] == [3, 1, 1, 1]


def test_an_out_of_memory_batch_is_retried_singly_and_the_cache_is_freed():
    module = load_app("parakeet", cuda=True)
    module._FFMPEG = None
    model = FakeParakeet()

    def reply(paths, kwargs):
        if len(paths) > 1:
            raise OutOfMemoryError("CUDA out of memory")
        return [hypothesis("ok")]

    model.reply = reply
    install_model(module, model)

    async def scenario():
        async with make_client(module) as client:
            return await post_batch(client, _batch_of(2))

    results = run(scenario()).json()["results"]

    assert [r["text"] for r in results] == ["ok", "ok"]
    assert ("empty_cache",) in module.torch.calls


def test_canary_batch_reports_a_model_failure_per_file():
    module = load_app("canary")
    module._FFMPEG = None
    model = FakeCanary()
    calls = []

    def reply(paths, kwargs):
        calls.append(paths)
        if len(calls) == 2:
            raise RuntimeError("bad audio")
        return [hypothesis("fine")]

    model.reply = reply
    install_model(module, model)

    async def scenario():
        async with make_client(module) as client:
            return await post_batch(client, _batch_of(3))

    results = run(scenario()).json()["results"]

    assert results[0]["text"] == "fine" and results[2]["text"] == "fine"
    assert results[1]["status"] == 500
    assert "bad audio" not in results[1]["error"] and "Request id:" in results[1]["error"]
    assert results[0]["language"] == "de"


# --- N3: canary languages -------------------------------------------------------------

def canary(monkeypatch, model=None, **env):
    return build("canary", monkeypatch, model=model, **env)


def decode(module, language=None, path="/transcribe", field="audio", **extra):
    async def scenario():
        async with make_client(module) as client:
            form = dict(extra)
            if language is not None:
                form["language"] = language
            return await post(client, wav_bytes(1.0), path=path, field=field, **form)

    return run(scenario())


def test_an_unsupported_language_is_a_422_and_the_model_is_never_run(monkeypatch):
    module, model = canary(monkeypatch)

    response = decode(module, "it")

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "'it'" in detail and "de, en, es, fr" in detail
    assert model.calls == [], "Italian audio used to be decoded as German"
    assert not module._model_slot.ever_loaded


def test_the_configured_checkpoint_decides_which_languages_are_accepted(monkeypatch):
    module, model = canary(monkeypatch, CANARY_ASR_MODEL="nvidia/canary-1b-v2")

    response = decode(module, "it")

    assert response.status_code == 200
    assert response.json()["language"] == "it"
    assert model.calls[0]["source_lang"] == "it" and model.calls[0]["target_lang"] == "it"
    assert len(module.SUPPORTED_LANGUAGES) == 25
    assert decode(module, "xx").status_code == 422


def test_the_override_replaces_the_derived_set(monkeypatch):
    module, model = canary(monkeypatch, CANARY_SUPPORTED_LANGUAGES="de, pl")

    assert decode(module, "pl").status_code == 200
    assert decode(module, "en").status_code == 422
    assert sorted(module.SUPPORTED_LANGUAGES) == ["de", "pl"]


@pytest.mark.parametrize("given, expected", [
    ("de-DE", "de"), ("de_DE", "de"), ("  EN ", "en"), ("Fr", "fr"),
])
def test_locale_tags_and_case_reduce_to_the_language(monkeypatch, given, expected):
    module, model = canary(monkeypatch)

    response = decode(module, given)

    assert response.status_code == 200
    assert response.json()["language"] == expected
    assert model.calls[0]["source_lang"] == expected


@pytest.mark.parametrize("given", ["auto", "AUTO", ""])
def test_auto_and_empty_take_the_default_language(monkeypatch, given):
    module, model = canary(monkeypatch, CANARY_DEFAULT_LANGUAGE="fr")

    response = decode(module, given)

    assert response.status_code == 200
    assert response.json()["language"] == "fr"


def test_no_language_field_takes_the_default(monkeypatch):
    module, model = canary(monkeypatch)

    assert decode(module).json()["language"] == "de"


def test_a_default_the_model_cannot_decode_falls_back_to_english_once_at_startup(monkeypatch):
    module, model = canary(monkeypatch, CANARY_DEFAULT_LANGUAGE="it")

    assert module.DEFAULT_LANGUAGE == "en"
    assert decode(module, "auto").json()["language"] == "en"


def test_the_openai_route_refuses_an_unsupported_language_too(monkeypatch):
    module, model = canary(monkeypatch)

    response = decode(module, "it", path="/v1/audio/transcriptions", field="file")

    assert response.status_code == 422 and model.calls == []
    assert decode(module, path="/v1/audio/transcriptions", field="file").status_code == 200


def test_a_batch_with_an_unsupported_language_is_refused_up_front(monkeypatch):
    module, model = canary(monkeypatch)

    async def scenario():
        async with make_client(module) as client:
            return await post_batch(client, _batch_of(2), language="it")

    response = run(scenario())

    assert response.status_code == 422 and model.calls == []


def test_status_lists_the_languages_of_the_configured_model(monkeypatch):
    module, _ = canary(monkeypatch, CANARY_ASR_MODEL="nvidia/canary-1b-v2")

    async def scenario():
        async with make_client(module) as client:
            return await client.get("/status")

    body = run(scenario()).json()

    assert "it" in body["supported_languages"] and len(body["supported_languages"]) == 25
    assert body["default_language"] == "de"


# --- canary: timestamps ---------------------------------------------------------------

def test_canary_asks_for_timestamps_and_the_language_by_default(monkeypatch):
    module, model = canary(monkeypatch)

    decode(module, "es")

    (call,) = model.calls
    assert call["timestamps"] is True
    assert (call["source_lang"], call["target_lang"], call["pnc"]) == ("es", "es", "yes")


def test_a_type_error_inside_the_decode_is_not_retried_and_not_hidden(monkeypatch):
    """The old `except TypeError` re-ran the whole decode without timestamps."""
    module, model = canary(monkeypatch)

    def explode(paths, kwargs):
        raise TypeError("unsupported operand type(s) for +: 'NoneType' and 'int'")

    model.reply = explode

    response = decode(module, "de")

    assert response.status_code == 500
    assert "unsupported operand" not in response.json()["detail"], "the internals stay in the log"
    assert len(model.calls) == 1, "the decode ran twice"


def test_a_transcribe_without_a_timestamps_parameter_is_called_without_it(monkeypatch):
    module, model = canary(monkeypatch, model=FakeCanaryWithoutTimestamps())

    response = decode(module, "de")

    assert response.status_code == 200
    assert "timestamps" not in model.calls[0]
    body = response.json()
    assert body["segments"] == [{"start": 0.0, "end": 1.0, "text": body["text"]}], \
        "without model timestamps the whole text becomes one segment"


def test_a_prompt_format_without_a_timestamp_slot_is_called_without_it(monkeypatch):
    """canary-1b rejects timestamps with a ValueError, which the old fallback never caught."""
    module, model = canary(monkeypatch, model=FakeCanaryV1())

    response = decode(module, "de")

    assert response.status_code == 200
    assert not model.calls[0]["timestamps"]  # the fake raises on a truthy one, like canary-1b
