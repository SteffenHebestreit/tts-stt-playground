"""HTTP behaviour of the Qwen3-ASR service.

Findings covered:

* qwen-asr generates at most ``max_new_tokens`` per piece it cuts (1200 s), and
  the service asked for 512, so speech beyond about two minutes came back as
  HTTP 200 with the end missing, as ONE segment 0..duration with confidence 1.0.
  Now the audio is cut into short pieces first: the whole transcript comes back,
  one segment per piece with its real start and end, and a piece that still hits
  the cap is flagged.
* an upload was read into memory whole with no cap and no length limit;
* ``ASR_MODEL_TTL=0`` still preloaded the model, the preload blocked start-up,
  and /unload took the slot lock on the event loop;
* the language table lacked eleven languages the model supports and
  full names ("German") fell through to auto-detect.

Every test drives the real app over ASGI with fakes only for torch, soundfile,
librosa, ffmpeg and the qwen_asr model (see ``test_qwen3_asr_harness``).
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from test_qwen3_asr_harness import (
    RATE, WAIT, FakeFfmpeg, encoded_wav, expected_transcript, fake_mp3, install_fake_qwen_asr,
    load_app, make_client, run, silent_wav, speech_levels, tmp_files, use_tmpdir, wait_for,
)

CHUNK_S = 30.0


def build(tmp_path, monkeypatch, *, ffmpeg=True, model_options=None, **env):
    """(app module, fake qwen_asr factory, fake ffmpeg or None, temp dir), no model loaded yet."""
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    use_tmpdir(monkeypatch, tmp)
    env.setdefault("ASR_MODEL_TTL", "-1")  # no idle timers left running after the test
    module = load_app(**env)
    factory = install_fake_qwen_asr(monkeypatch, **(model_options or {}))
    fake = None
    if ffmpeg:
        fake = FakeFfmpeg(tmp_path)
        fake.install(module, monkeypatch)
    else:
        monkeypatch.setattr(module, "_FFMPEG", None, raising=False)
    return module, factory, fake, tmp


def load_model(module):
    """Load the model through the app's own loader (and so through the fake qwen_asr)."""
    with module._model_slot.acquire() as model:
        return model


async def post(client, data, *, name="clip.wav", path="/transcribe", field="audio", **form):
    return await client.post(path, files={field: (name, data, "audio/wav")}, data=form)


def transcribe(module, data, **form):
    async def scenario():
        async with make_client(module) as client:
            return await post(client, data, **form)
    return run(scenario())


# --- the truncation and the fake segment ----------------------------------------------

def test_a_long_recording_is_transcribed_in_full_not_cut_off_after_two_minutes(tmp_path, monkeypatch):
    module, factory, _, _ = build(tmp_path, monkeypatch)
    levels = speech_levels(1200)  # ~330 s at 4 words per second
    r = transcribe(module, encoded_wav(levels))
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == expected_transcript(levels)
    assert body["truncated"] is False
    assert body["warnings"] == []
    assert body["duration"] == pytest.approx(len(levels) * 0.25)


def test_each_piece_gets_its_own_segment_with_the_real_start_and_end(tmp_path, monkeypatch):
    module, factory, _, _ = build(tmp_path, monkeypatch)
    levels = speech_levels(1200)
    body = transcribe(module, encoded_wav(levels)).json()
    segments = body["segments"]
    duration = body["duration"]

    assert len(segments) == body["chunks"] >= 11
    assert segments[0]["start"] == 0.0
    assert segments[-1]["end"] == pytest.approx(duration, abs=0.001)
    for previous, current in zip(segments, segments[1:]):
        assert current["start"] == previous["end"]
    assert all(0 < s["end"] - s["start"] <= CHUNK_S + 0.001 for s in segments)
    # Together the segments are the transcript, each holding only its own words.
    assert " ".join(s["text"] for s in segments) == body["text"]
    assert all(0 < len(s["text"].split()) <= 120 for s in segments)  # a piece holds ~100 words, not 1200


def test_segments_carry_no_invented_confidence(tmp_path, monkeypatch):
    module, *_ = build(tmp_path, monkeypatch)
    body = transcribe(module, encoded_wav(speech_levels(30))).json()
    assert body["segments"]
    assert all("confidence" not in s for s in body["segments"])
    assert body["language_probability"] is None


def test_the_model_is_given_short_pieces_not_the_file(tmp_path, monkeypatch):
    module, factory, _, _ = build(tmp_path, monkeypatch)
    levels = speech_levels(1200)
    transcribe(module, encoded_wav(levels))
    (call,) = factory.model.calls  # one transcribe() call: the library batches the pieces itself
    assert call["items"] > 1
    assert not any(call["as_paths"])
    assert set(call["rates"]) == {RATE}
    assert all(seconds <= CHUNK_S + 1e-6 for seconds in call["lengths_s"])
    assert sum(call["lengths_s"]) == pytest.approx(len(levels) * 0.25, abs=0.001)


def test_a_short_recording_is_one_piece_and_one_segment(tmp_path, monkeypatch):
    module, factory, _, _ = build(tmp_path, monkeypatch)
    levels = speech_levels(20)
    body = transcribe(module, encoded_wav(levels)).json()
    assert body["chunks"] == 1
    assert body["text"] == expected_transcript(levels)
    (segment,) = body["segments"]
    assert (segment["start"], segment["end"]) == (0.0, pytest.approx(len(levels) * 0.25))


def test_the_piece_length_is_configurable(tmp_path, monkeypatch):
    module, factory, _, _ = build(tmp_path, monkeypatch, QWEN3_ASR_CHUNK_S="10")
    levels = speech_levels(200)  # 55 s
    body = transcribe(module, encoded_wav(levels)).json()
    assert body["text"] == expected_transcript(levels)
    assert body["chunks"] >= 6
    assert all(seconds <= 10.0 + 1e-6 for seconds in factory.model.calls[0]["lengths_s"])


def test_silence_between_speech_produces_no_empty_segments(tmp_path, monkeypatch):
    module, *_ = build(tmp_path, monkeypatch)
    levels = speech_levels(20) + [0] * 400 + speech_levels(20)  # 100 s of silence in the middle
    body = transcribe(module, encoded_wav(levels)).json()
    assert body["text"] == " ".join([expected_transcript(speech_levels(20)), expected_transcript(speech_levels(20))])
    assert body["chunks"] >= 4 and len(body["segments"]) == 2  # the silent pieces made no segment
    assert all(s["text"] for s in body["segments"])


def test_a_recording_with_no_speech_is_an_empty_result_not_an_error(tmp_path, monkeypatch):
    module, *_ = build(tmp_path, monkeypatch)
    r = transcribe(module, silent_wav(3.0))
    assert r.status_code == 200
    assert r.json()["text"] == "" and r.json()["segments"] == [] and r.json()["language"] == ""


# --- the output limit ----------------------------------------------------------------------

def test_the_default_output_limit_is_far_above_what_a_piece_can_need(tmp_path, monkeypatch):
    module, factory, _, _ = build(tmp_path, monkeypatch)
    load_model(module)
    (load,) = factory.loads
    # A very fast speaker makes ~10 tokens per second; 30 s is 300. It was 512 for
    # a piece of up to 20 minutes.
    assert load["max_new_tokens"] == module.MAX_NEW_TOKENS >= 20 * CHUNK_S
    assert load["max_inference_batch_size"] == module.BATCH_SIZE == 4


def test_a_piece_that_stops_on_the_output_limit_is_flagged(tmp_path, monkeypatch):
    module, factory, _, _ = build(tmp_path, monkeypatch, QWEN3_ASR_MAX_NEW_TOKENS="40")
    levels = speech_levels(100)  # 27.5 s: one piece with 100 words, the model can say 37
    body = transcribe(module, encoded_wav(levels)).json()
    assert body["truncated"] is True
    (segment,) = body["segments"]
    assert segment["truncated"] is True
    assert len(segment["text"].split()) < 100
    (warning,) = body["warnings"]
    assert "QWEN3_ASR_MAX_NEW_TOKENS" in warning and "QWEN3_ASR_CHUNK_S" in warning


def test_a_piece_below_the_limit_is_not_flagged(tmp_path, monkeypatch):
    module, *_ = build(tmp_path, monkeypatch, QWEN3_ASR_MAX_NEW_TOKENS="40")
    body = transcribe(module, encoded_wav(speech_levels(20))).json()
    assert body["truncated"] is False
    assert "truncated" not in body["segments"][0]


# --- upload and duration limits -----------------------------------------------------------

def test_an_upload_over_the_limit_is_a_413_and_the_model_never_runs(tmp_path, monkeypatch):
    module, factory, _, tmp = build(tmp_path, monkeypatch, MAX_UPLOAD_MB="0.05")  # 52 KB
    r = transcribe(module, encoded_wav(speech_levels(100)))  # ~800 KB: past the file limit, within the body slack
    assert r.status_code == 413
    assert "MAX_UPLOAD_MB" in r.json()["detail"]
    assert factory.loads == []
    assert tmp_files(tmp) == []


def test_a_huge_body_is_refused_from_its_header_before_it_is_parsed(tmp_path, monkeypatch):
    module, factory, _, tmp = build(tmp_path, monkeypatch, MAX_UPLOAD_MB="0.05")
    r = transcribe(module, b"\0" * (3 * 1024 * 1024))
    assert r.status_code == 413
    assert "upload limit" in r.json()["detail"]
    assert factory.loads == [] and tmp_files(tmp) == []


def test_an_upload_inside_the_limit_is_accepted(tmp_path, monkeypatch):
    module, *_ = build(tmp_path, monkeypatch, MAX_UPLOAD_MB="0.1")
    assert transcribe(module, encoded_wav(speech_levels(10))).status_code == 200


def test_audio_longer_than_the_limit_is_a_413_before_the_model_is_touched(tmp_path, monkeypatch):
    module, factory, _, tmp = build(tmp_path, monkeypatch, MAX_AUDIO_SECONDS="10")
    r = transcribe(module, silent_wav(20.0))
    assert r.status_code == 413
    assert "MAX_AUDIO_SECONDS" in r.json()["detail"]
    assert factory.loads == [] and tmp_files(tmp) == []
    assert transcribe(module, encoded_wav(speech_levels(30))).status_code == 200  # 8.25 s


def test_a_header_that_understates_the_length_cannot_get_past_the_limit(tmp_path, monkeypatch):
    module, factory, fake, tmp = build(tmp_path, monkeypatch, MAX_AUDIO_SECONDS="10")
    r = transcribe(module, fake_mp3(actual_seconds=3600, declared_seconds=5), name="clip.mp3")
    assert r.status_code == 413
    (call,) = fake.calls()
    # ffmpeg was told to stop just past the limit, so an hour was never decoded.
    assert float(call["args"][call["args"].index("-t") + 1]) == pytest.approx(11.0)
    assert call["wrote_seconds"] == pytest.approx(11.0)
    assert factory.loads == [] and tmp_files(tmp) == []


def test_a_zero_limit_means_no_limit(tmp_path, monkeypatch):
    module, *_ = build(tmp_path, monkeypatch, MAX_AUDIO_SECONDS="0")
    assert transcribe(module, silent_wav(20.0)).status_code == 200


def test_the_python_decoder_is_bounded_too_when_ffmpeg_is_missing(tmp_path, monkeypatch):
    module, *_ = build(tmp_path, monkeypatch, ffmpeg=False, MAX_AUDIO_SECONDS="10")
    assert transcribe(module, silent_wav(20.0)).status_code == 413  # by its header, before any decode
    assert module._test_librosa.load_calls == []
    assert transcribe(module, encoded_wav(speech_levels(30))).status_code == 200
    (call,) = module._test_librosa.load_calls
    assert call["duration"] == pytest.approx(11.0) and call["sr"] == RATE and call["mono"] is True


def test_an_empty_upload_is_a_400(tmp_path, monkeypatch):
    module, factory, _, tmp = build(tmp_path, monkeypatch)
    r = transcribe(module, b"")
    assert r.status_code == 400
    assert tmp_files(tmp) == []


def test_a_file_that_is_not_audio_is_a_422_and_leaves_no_temp_file(tmp_path, monkeypatch):
    module, factory, _, tmp = build(tmp_path, monkeypatch)
    r = transcribe(module, b"this is not audio", name="clip.wav")
    assert r.status_code == 422
    assert factory.loads == [] and tmp_files(tmp) == []


def test_ffmpeg_failing_falls_back_to_the_python_decoder(tmp_path, monkeypatch):
    module, factory, fake, tmp = build(tmp_path, monkeypatch)
    fake.mode = "fail"
    levels = speech_levels(40)
    r = transcribe(module, encoded_wav(levels))
    assert r.status_code == 200 and r.json()["text"] == expected_transcript(levels)
    assert module._test_librosa.load_calls
    assert tmp_files(tmp) == []


def test_temp_files_are_removed_after_every_outcome(tmp_path, monkeypatch):
    module, factory, _, tmp = build(tmp_path, monkeypatch, model_options={"error": RuntimeError("boom")})
    assert transcribe(module, encoded_wav(speech_levels(10))).status_code == 500
    assert tmp_files(tmp) == []


# --- errors ----------------------------------------------------------------------------------

def test_gpu_out_of_memory_is_a_503_that_says_what_to_change(tmp_path, monkeypatch):
    error = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    module, *_ = build(tmp_path, monkeypatch, model_options={"error": error})
    r = transcribe(module, encoded_wav(speech_levels(10)))
    assert r.status_code == 503
    assert "QWEN3_ASR_BATCH_SIZE" in r.json()["detail"]


def test_any_other_model_failure_is_a_500_with_the_message(tmp_path, monkeypatch):
    module, *_ = build(tmp_path, monkeypatch, model_options={"error": RuntimeError("weights are corrupt")})
    r = transcribe(module, encoded_wav(speech_levels(10)))
    assert r.status_code == 500
    assert "weights are corrupt" in r.json()["detail"]


# --- languages ----------------------------------------------------------------------------------

@pytest.mark.parametrize("requested,sent", [
    ("de", "German"), ("DE", "German"), ("de-DE", "German"), ("de_CH", "German"),
    ("German", "German"), ("german", "German"), ("  de  ", "German"),
    ("en", "English"), ("pt-BR", "Portuguese"), ("zh", "Chinese"), ("yue", "Cantonese"),
    ("sv", "Swedish"), ("da", "Danish"), ("fi", "Finnish"), ("pl", "Polish"), ("cs", "Czech"),
    ("fil", "Filipino"), ("tl", "Filipino"), ("fa", "Persian"), ("el", "Greek"),
    ("ro", "Romanian"), ("hu", "Hungarian"), ("mk", "Macedonian"),
    ("auto", None), ("AUTO", None), ("", None),
])
def test_the_request_language_reaches_the_model_as_a_name_it_accepts(tmp_path, monkeypatch, requested, sent):
    module, factory, _, _ = build(tmp_path, monkeypatch)
    body = transcribe(module, encoded_wav(speech_levels(10)), language=requested).json()
    assert factory.model.calls[0]["language"] == sent
    assert body["warnings"] == []
    if sent:
        assert body["language"] == sent


def test_every_language_the_table_offers_is_one_qwen_asr_supports():
    module = load_app()
    # qwen-asr 0.0.6 utils.SUPPORTED_LANGUAGES
    supported = {
        "Chinese", "English", "Cantonese", "Arabic", "German", "French", "Spanish", "Portuguese",
        "Indonesian", "Italian", "Korean", "Russian", "Thai", "Vietnamese", "Japanese", "Turkish",
        "Hindi", "Malay", "Dutch", "Swedish", "Danish", "Finnish", "Polish", "Czech", "Filipino",
        "Persian", "Greek", "Romanian", "Hungarian", "Macedonian",
    }
    assert set(module.LANGUAGE_NAME_MAP.values()) == supported


def test_a_language_the_model_does_not_know_falls_back_to_detection_and_says_so(tmp_path, monkeypatch):
    module, factory, _, _ = build(tmp_path, monkeypatch)
    body = transcribe(module, encoded_wav(speech_levels(10)), language="uk").json()
    assert factory.model.calls[0]["language"] is None
    assert any("'uk'" in w and "automatically" in w for w in body["warnings"])
    assert body["language"] == "German"  # what the model detected


def test_the_file_language_is_the_one_most_of_the_audio_was_in(tmp_path, monkeypatch):
    module, *_ = build(
        tmp_path, monkeypatch, model_options={"piece_languages": ["German", "German", "English"]},
    )
    body = transcribe(module, encoded_wav(speech_levels(400))).json()  # 110 s: four pieces
    languages = [s["language"] for s in body["segments"]]
    assert "English" in languages and languages.count("German") > languages.count("English")
    assert body["language"] == "German"


# --- detect_language --------------------------------------------------------------------------------

def test_language_detection_runs_only_the_first_piece_of_a_long_file(tmp_path, monkeypatch):
    module, factory, _, tmp = build(tmp_path, monkeypatch)
    levels = speech_levels(1200)

    async def scenario():
        async with make_client(module) as client:
            return await post(client, encoded_wav(levels), path="/detect_language", field="file")

    r = run(scenario())
    assert r.status_code == 200
    body = r.json()
    assert body["detected_language"] == "German"
    assert body["sample_text"] == expected_transcript(levels)[:200]
    assert body["audio_duration"] == pytest.approx(len(levels) * 0.25)
    (call,) = factory.model.calls
    assert call["items"] == 1 and not any(call["as_paths"])
    assert call["lengths_s"][0] <= CHUNK_S + 1e-6
    assert tmp_files(tmp) == []


def test_language_detection_on_a_file_without_a_header_length_uses_the_decoded_length(tmp_path, monkeypatch):
    module, *_ = build(tmp_path, monkeypatch)

    async def scenario():
        async with make_client(module) as client:
            return await post(client, fake_mp3(actual_seconds=4, declared_seconds=None), name="a.mp3",
                              path="/detect_language", field="file")

    body = run(scenario()).json()
    assert body["audio_duration"] == pytest.approx(4.0)


def test_language_detection_honours_the_upload_limit(tmp_path, monkeypatch):
    module, factory, _, _ = build(tmp_path, monkeypatch, MAX_UPLOAD_MB="0.05")

    async def scenario():
        async with make_client(module) as client:
            return await post(client, encoded_wav(speech_levels(100)), path="/detect_language", field="file")

    assert run(scenario()).status_code == 413
    assert factory.loads == []


# --- batch --------------------------------------------------------------------------------------------

def test_a_batch_reports_each_file_and_one_bad_file_does_not_fail_the_rest(tmp_path, monkeypatch):
    module, factory, _, tmp = build(tmp_path, monkeypatch, MAX_UPLOAD_MB="0.1")
    small = speech_levels(8)  # 2 s: 64 KB, under the 0.1 MB limit

    async def scenario():
        async with make_client(module) as client:
            files = [
                ("audios", ("a.wav", encoded_wav(small), "audio/wav")),
                ("audios", ("too-big.wav", encoded_wav(speech_levels(100)), "audio/wav")),
                ("audios", ("junk.wav", b"not audio", "audio/wav")),
                ("audios", ("b.wav", encoded_wav(small), "audio/wav")),
            ]
            return await client.post("/transcribe-batch", files=files, data={"language": "de"})

    r = run(scenario())
    assert r.status_code == 200
    body = r.json()
    assert body["batch"] is True and body["file_count"] == 4
    a, big, junk, b = body["results"]
    assert a["filename"] == "a.wav" and a["text"] == expected_transcript(small)
    assert a["segments"] and a["language"] == "German" and a["duration"] > 0
    assert big["filename"] == "too-big.wav" and big["status"] == 413 and "text" not in big
    assert junk["status"] == 422
    assert b["text"] == expected_transcript(small)
    assert tmp_files(tmp) == []


# --- model loading ---------------------------------------------------------------------------------------

def test_the_loader_passes_the_configured_limits_to_the_model(tmp_path, monkeypatch):
    module, factory, _, _ = build(
        tmp_path, monkeypatch, QWEN3_ASR_MODEL="Qwen/Qwen3-ASR-0.6B", QWEN3_ASR_BATCH_SIZE="4",
        QWEN3_ASR_MAX_NEW_TOKENS="900",
    )
    load_model(module)
    (load,) = factory.loads
    assert load["name"] == "Qwen/Qwen3-ASR-0.6B"
    assert load["max_inference_batch_size"] == 4 and load["max_new_tokens"] == 900
    assert load["device_map"] == "cpu" and load["dtype"] == "float32"
    assert "attn_implementation" not in load


def test_the_output_limit_follows_the_piece_length(tmp_path, monkeypatch):
    module, factory, _, _ = build(tmp_path, monkeypatch, QWEN3_ASR_CHUNK_S="120")
    assert module.CHUNK_S == 120.0 and module.MAX_NEW_TOKENS == 2400


@pytest.mark.parametrize("value,expected", [("abc", 30.0), ("2", 30.0), ("9999", 600.0), ("15", 15.0)])
def test_a_bad_piece_length_cannot_stop_the_service_from_starting(tmp_path, monkeypatch, value, expected):
    module, *_ = build(tmp_path, monkeypatch, QWEN3_ASR_CHUNK_S=value)
    assert module.CHUNK_S == expected


@pytest.mark.parametrize("value,passed", [
    ("sdpa", "sdpa"), ("EAGER", "eager"), ("flash_attention_2", "flash_attention_2"),
    ("", None), ("turbo", None),
])
def test_the_attention_implementation_is_only_passed_when_asked_for(tmp_path, monkeypatch, value, passed):
    module, factory, _, _ = build(tmp_path, monkeypatch, QWEN3_ASR_ATTN_IMPLEMENTATION=value)
    load_model(module)
    assert factory.loads[0].get("attn_implementation") == passed


# --- start-up, readiness, unload ---------------------------------------------------------------------------

def test_ttl_zero_does_not_load_the_model_at_start_up(tmp_path, monkeypatch):
    module, factory, _, _ = build(tmp_path, monkeypatch, ASR_MODEL_TTL="0")

    async def scenario():
        async with module.app.router.lifespan_context(module.app):
            assert factory.loads == []
            assert getattr(module, "_preload_pending", False) is False
            await asyncio.sleep(0)
        return factory.loads

    assert run(scenario()) == []


def test_the_preload_runs_in_the_background_and_ready_says_loading_meanwhile(tmp_path, monkeypatch):
    module, factory, _, _ = build(tmp_path, monkeypatch)
    factory.block = threading.Event()

    async def scenario():
        async with make_client(module) as client:
            # The old lifespan loaded the model before `yield`, so this line was
            # never reached until the download finished: the port stayed closed.
            async with module.app.router.lifespan_context(module.app):
                await wait_for(factory.entered.is_set, "the preload to start")
                ready = await client.get("/ready")
                health = await client.get("/health")
                assert ready.status_code == 503
                assert ready.json()["reason"] == "loading" and ready.headers["retry-after"] == "5"
                assert health.status_code == 200 and health.json()["model_resident"] is False

                factory.block.set()
                await wait_for(lambda: module._model_slot.resident, "the model to load")
                await wait_for(lambda: not module._preload_pending, "the preload to finish")
                ready = await client.get("/ready")
                assert ready.status_code == 200
                assert ready.json()["ready"] is True and ready.json()["model_ever_loaded"] is True

    run(scenario())


def test_ready_without_a_preload_is_200_and_does_not_load_the_model(tmp_path, monkeypatch):
    module, factory, _, _ = build(tmp_path, monkeypatch)

    async def scenario():
        async with make_client(module) as client:
            return await client.get("/ready"), await client.get("/health")

    ready, health = run(scenario())
    assert ready.status_code == 200 and ready.json()["reason"] == "ok"
    assert ready.json()["current_model"] == "Qwen/Qwen3-ASR-1.7B"
    assert health.status_code == 200
    assert factory.loads == []


def test_ready_reports_a_failed_first_load_with_the_reason(tmp_path, monkeypatch):
    module, factory, _, _ = build(tmp_path, monkeypatch)
    factory.fail = RuntimeError("no such checkpoint")

    async def scenario():
        async with make_client(module) as client:
            with pytest.raises(RuntimeError):
                await asyncio.to_thread(module.get_model)
            failed = await client.get("/ready")
            factory.fail = None
            await asyncio.to_thread(module.get_model)
            ok = await client.get("/ready")
            return failed, ok

    failed, ok = run(scenario())
    assert failed.status_code == 503
    assert failed.json()["reason"] == "load_failed" and "no such checkpoint" in failed.json()["detail"]
    assert ok.status_code == 200 and ok.json()["model_ever_loaded"] is True


def test_unload_does_not_freeze_the_event_loop_while_the_weights_are_released(tmp_path, monkeypatch):
    module, factory, _, _ = build(tmp_path, monkeypatch)
    entered, release, timed_out = threading.Event(), threading.Event(), []

    def slow_release(model):
        entered.set()
        if not release.wait(WAIT):  # a blocked event loop cannot get to release.set()
            timed_out.append(True)

    module._model_slot = module.ModelSlot(
        module._load_qwen3_asr, ttl_seconds=-1, name="test", on_unload=slow_release,
    )

    async def scenario():
        async with make_client(module) as client:
            await asyncio.to_thread(module.get_model)
            unloading = asyncio.create_task(client.post("/unload"))
            await wait_for(entered.is_set, "the release hook to start")
            health = await asyncio.wait_for(client.get("/health"), WAIT)
            assert health.status_code == 200
            assert not timed_out, "the event loop was blocked while the model was being released"
            release.set()
            return await unloading

    r = run(scenario())
    assert r.status_code == 200 and r.json()["model_resident"] is False


def test_unload_while_a_request_is_in_flight_is_a_409(tmp_path, monkeypatch):
    gate = threading.Event()
    module, factory, _, _ = build(tmp_path, monkeypatch, model_options={"gate": gate})

    async def scenario():
        async with make_client(module) as client:
            pending = asyncio.create_task(post(client, encoded_wav(speech_levels(10))))
            await wait_for(lambda: factory.models and factory.model.entered.is_set(), "the model to be running")
            busy = await client.post("/unload")
            gate.set()
            done = await pending
            idle = await client.post("/unload")
            return busy, done, idle

    busy, done, idle = run(scenario())
    assert busy.status_code == 409 and busy.json()["reason"] == "busy"
    assert done.status_code == 200
    assert idle.status_code == 200 and idle.json()["model_resident"] is False


def test_status_reports_the_limits_in_force(tmp_path, monkeypatch):
    module, *_ = build(tmp_path, monkeypatch, MAX_UPLOAD_MB="12", MAX_AUDIO_SECONDS="600")

    async def scenario():
        async with make_client(module) as client:
            return await client.get("/status")

    body = run(scenario()).json()
    assert body["max_upload_mb"] == 12 and body["max_audio_seconds"] == 600
    assert body["chunk_seconds"] == 30 and body["batch_size"] == 4 and body["max_new_tokens"] == 600
    assert body["model_resident"] is False


def test_a_bad_concurrency_setting_cannot_stop_the_service_from_starting(tmp_path, monkeypatch):
    module, *_ = build(tmp_path, monkeypatch, ASR_MAX_CONCURRENCY="lots")
    assert module._ASR_SEM._value == 1
