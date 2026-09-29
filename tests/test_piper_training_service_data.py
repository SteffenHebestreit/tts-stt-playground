"""Data path of the piper-training-service: what it reads, what language it assumes, what it trusts.

Four findings, all in the code between "the client sent something" and "a
training dataset exists":

* `/prepare-dataset` opened whatever `audio_path` named, and fetched anything
  starting with "http" with an unbounded GET. Any file the process can read, and
  any host its network can reach, was one request away.
* The language never reached the phonemiser: German recordings were phonemised as
  `en-us`, and the STT call was hard-wired to `auto`.
* The STT confidence filter compared a value that was always 1.0, because
  stt-service reports `avg_logprob`, not `confidence`.
* Errors were turned into empty results: an unreachable STT service ended as
  "no valid segments", a phonemiser failure as raw text used as phonemes.

Driven through the real endpoints and modules (see the support module); the audio
and network libraries are the only stand-ins.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_piper_training_service_support import (  # noqa: F401  (training_service is a fixture)
    asgi_client, install_fake_stt, make_dataset, settle, status_of, stt_result, training_service, write_job_state,
    write_wav,
)


def segment(path, text="das ist ein satz", start=0.0, end=1.5):
    return {"audio_path": str(path), "text": text, "start_time": start, "end_time": end}


def prepare(client, model="voice", segments=(), **extra):
    return client.post("/prepare-dataset", json={"model_name": model, "segments": list(segments), **extra})


def clips(root: Path, count: int = 3, folder: str = "raw") -> list:
    """WAV files inside the data directory, as an earlier upload would leave them."""
    return [write_wav(root / "data" / "voice" / folder / f"clip_{i}.wav", seconds=2.0) for i in range(count)]


# --- where audio may come from ---------------------------------------------------------------


class Spy:
    """Records what the service tried to read or fetch."""

    def __init__(self, svc, monkeypatch):
        self.fetched: list = []
        self.opened: list = []
        spy = self
        original_load = svc.librosa.load

        def load(path, *args, **kwargs):
            spy.opened.append((str(path) if not hasattr(path, "read") else "<bytes>", kwargs))
            return original_load(path, *args, **kwargs)

        monkeypatch.setattr(svc.librosa, "load", load)

        class Session(svc.aiohttp.ClientSession):
            def get(self, url, **kwargs):
                spy.fetched.append((url, kwargs))
                return super().get(url, **kwargs)

        monkeypatch.setattr(svc.aiohttp, "ClientSession", Session)


def test_urls_are_not_fetched_by_default(training_service, monkeypatch):
    svc = training_service()
    spy = Spy(svc, monkeypatch)

    async def run():
        async with asgi_client(svc.module) as client:
            for url in ("http://169.254.169.254/latest/meta-data/", "https://internal.corp/a.wav",
                        "HTTP://Internal.Corp/a.wav", "ftp://host/a.wav", "file:///etc/passwd"):
                response = await prepare(client, segments=[segment(url), segment(url)])
                assert response.status_code == 400, f"{url}: {response.status_code} {response.text}"
            assert spy.fetched == [], f"the service fetched {spy.fetched}"
            assert spy.opened == []

    asyncio.run(run())


@pytest.mark.parametrize("path", [
    "/etc/passwd", "../secrets.wav", "data/../../etc/passwd", "data/voice/../../outside.wav",
    "checkpoints/x/final_model.pt", "/proc/self/environ",
])
def test_paths_outside_the_data_directory_are_not_read(training_service, monkeypatch, path):
    svc = training_service()
    spy = Spy(svc, monkeypatch)

    async def run():
        async with asgi_client(svc.module) as client:
            response = await prepare(client, segments=[segment(path), segment(path)])
            assert response.status_code == 400, response.text
            assert spy.opened == [], f"the service opened {spy.opened}"
            assert not (svc.root / "data" / "voice").exists(), "a rejected request created a dataset"

    asyncio.run(run())


def test_a_symlink_inside_data_cannot_lead_outside_it(training_service, monkeypatch):
    svc = training_service()
    outside = write_wav(svc.root / "outside" / "secret.wav")
    link = svc.root / "data" / "voice" / "raw" / "innocent.wav"
    link.parent.mkdir(parents=True)
    link.symlink_to(outside)
    spy = Spy(svc, monkeypatch)

    async def run():
        async with asgi_client(svc.module) as client:
            response = await prepare(client, segments=[segment(link), segment(link)])
            assert response.status_code == 400
            assert spy.opened == []

    asyncio.run(run())


def test_relative_and_absolute_paths_inside_data_are_accepted(training_service):
    svc = training_service()
    files = clips(svc.root, 3)

    async def run():
        async with asgi_client(svc.module) as client:
            response = await prepare(client, segments=[
                segment("data/voice/raw/clip_0.wav"), segment(files[1]), segment(files[2])])
            assert response.status_code == 200, response.text
            assert response.json()["num_samples"] == 3

    asyncio.run(run())


def test_extra_directories_can_be_allowed_explicitly(training_service):
    svc = training_service()
    inbox = svc.root / "inbox"
    files = [write_wav(inbox / f"in_{i}.wav") for i in range(2)]

    async def run():
        async with asgi_client(svc.module) as client:
            assert (await prepare(client, segments=[segment(f) for f in files])).status_code == 400
        svc2 = training_service({"TRAINING_ALLOWED_AUDIO_DIRS": str(inbox)})
        async with asgi_client(svc2.module) as client:
            response = await prepare(client, segments=[segment(f) for f in files])
            assert response.status_code == 200, response.text

    asyncio.run(run())


class FakeDownload:
    """Stands in for aiohttp's response to session.get()."""

    def __init__(self, body: bytes, status: int = 200, content_length="auto"):
        self.body, self.status = body, status
        self.content_length = len(body) if content_length == "auto" else content_length
        self.content = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    async def read(self):
        return self.body

    async def iter_chunked(self, size):
        for start in range(0, len(self.body), size):
            yield self.body[start:start + size]


def wav_bytes(tmp: Path) -> bytes:
    return write_wav(tmp / "dl.wav", seconds=2.0).read_bytes()


def test_an_allowed_host_is_fetched_without_redirects_and_within_the_size_limit(training_service, monkeypatch):
    svc = training_service({"TRAINING_ALLOWED_URL_HOSTS": "files.example.com"})
    body = wav_bytes(svc.root)
    requests: list = []

    class Session(svc.aiohttp.ClientSession):
        def get(self, url, **kwargs):
            requests.append((url, kwargs))
            return FakeDownload(body)

    monkeypatch.setattr(svc.aiohttp, "ClientSession", Session)
    url = "https://files.example.com/a/clip.wav"

    async def run():
        async with asgi_client(svc.module) as client:
            response = await prepare(client, segments=[segment(url), segment(url)])
            assert response.status_code == 200, response.text
            assert response.json()["num_samples"] == 2
            assert [u for u, _ in requests] == [url, url]
            assert all(kw.get("allow_redirects") is False for _, kw in requests), (
                "a redirect would let an allowed host send the request to one that is not")

    asyncio.run(run())


@pytest.mark.parametrize("url", [
    "https://other.example.com/a.wav",
    "https://files.example.com.evil.net/a.wav",
    "https://files.example.com@evil.net/a.wav",
])
def test_only_the_listed_host_is_fetched(training_service, monkeypatch, url):
    svc = training_service({"TRAINING_ALLOWED_URL_HOSTS": "files.example.com"})
    spy = Spy(svc, monkeypatch)

    async def run():
        async with asgi_client(svc.module) as client:
            response = await prepare(client, segments=[segment(url), segment(url)])
            assert response.status_code == 400
            assert spy.fetched == []

    asyncio.run(run())


@pytest.mark.parametrize("declared,status", [
    ("exact", 200),   # Content-Length says it is too big
    (None, 200),      # no Content-Length: only counting the stream can tell
    ("exact", 302),   # a redirect, which an allowed host could use to reach one that is not
])
def test_a_download_is_bounded_and_redirects_are_refused(training_service, monkeypatch, declared, status):
    svc = training_service({"TRAINING_ALLOWED_URL_HOSTS": "files.example.com", "MAX_UPLOAD_MB": "0.01"})
    body = wav_bytes(svc.root)  # a perfectly good recording, well over 10 KiB
    assert len(body) > 10 * 1024

    class Session(svc.aiohttp.ClientSession):
        def get(self, url, **kwargs):
            return FakeDownload(body, status=status, content_length="auto" if declared else None)

    monkeypatch.setattr(svc.aiohttp, "ClientSession", Session)
    url = "https://files.example.com/big.wav"

    async def run():
        async with asgi_client(svc.module) as client:
            result = await prepare(client, segments=[segment(url), segment(url)])
            # Every segment failed to load, so there is nothing to build a dataset from.
            assert result.status_code >= 400, "an over-size or redirected download was used"
            assert list((svc.root / "data" / "voice" / "audio").glob("*.wav")) == []

    asyncio.run(run())


# --- the request's language reaches the phonemiser ---------------------------------------------------


@pytest.mark.parametrize("requested,voice", [
    (None, "de"), ("de", "de"), ("de-AT", "de"), ("en", "en-us"), ("en_GB", "en-gb"), ("fr", "fr-fr"),
])
def test_prepare_dataset_phonemises_in_the_requested_language(training_service, requested, voice):
    svc = training_service()
    files = clips(svc.root)
    extra = {} if requested is None else {"language": requested}

    async def run():
        async with asgi_client(svc.module) as client:
            response = await prepare(client, segments=[segment(f) for f in files], **extra)
            assert response.status_code == 200, response.text
            assert {c.language for c in svc.phonemizer.calls} == {voice}, (
                f"language={requested!r} was phonemised as {[c.language for c in svc.phonemizer.calls]}")

    asyncio.run(run())


def test_the_configured_default_language_applies_when_none_is_given(training_service):
    svc = training_service({"TRAINING_DEFAULT_LANGUAGE": "fr"})
    files = clips(svc.root)

    async def run():
        async with asgi_client(svc.module) as client:
            response = await prepare(client, segments=[segment(f) for f in files])
            assert response.status_code == 200
            assert {c.language for c in svc.phonemizer.calls} == {"fr-fr"}
            assert response.json()["language"] == "fr"

    asyncio.run(run())


def test_an_unsupported_language_is_refused_not_phonemised_as_english(training_service):
    svc = training_service()
    files = clips(svc.root)

    async def run():
        async with asgi_client(svc.module) as client:
            response = await prepare(client, segments=[segment(f) for f in files], language="klingon")
            assert response.status_code == 400
            assert "Supported" in response.json()["detail"]
            assert svc.phonemizer.calls == []

    asyncio.run(run())


def test_a_bad_default_language_stops_the_service_at_startup(training_service):
    with pytest.raises(RuntimeError, match="TRAINING_DEFAULT_LANGUAGE"):
        training_service({"TRAINING_DEFAULT_LANGUAGE": "klingon"})


# --- a phonemiser failure is never papered over with raw text -----------------------------------------------


def make_segments(root: Path, texts):
    from audio_segmenter import TrainingSegment

    out = []
    for i, text in enumerate(texts):
        out.append(TrainingSegment(
            audio_path=root / "data" / "voice" / "audio" / f"voice_a_{i:04d}.wav",
            text=text, duration=2.0, original_file="a.wav"))
    return out


def segmenter(monkeypatch):
    import audio_segmenter

    monkeypatch.setattr(audio_segmenter.AudioSegmenter, "_find_ffmpeg", lambda self: "ffmpeg")
    return audio_segmenter.AudioSegmenter()


def fail_on(svc, marker: str):
    real = svc.phonemizer.phonemize

    def phonemize(text, *args, **kwargs):
        if marker in text:
            raise RuntimeError("espeak could not pronounce this")
        return real(text, *args, **kwargs)

    svc.phonemizer.phonemize = phonemize


def test_a_sample_that_cannot_be_phonemised_is_dropped_and_counted_never_kept_as_text(
        training_service, monkeypatch):
    svc = training_service()
    fail_on(svc, "BOOM")
    texts = [f"satz nummer {i}" for i in range(40)] + ["BOOM klingt schlecht"]
    (svc.root / "data" / "voice").mkdir(parents=True)
    seg = segmenter(monkeypatch)

    seg.generate_training_metadata(make_segments(svc.root, texts), svc.root / "data" / "voice", "voice", "de")

    entries = json.loads((svc.root / "data" / "voice" / "metadata.json").read_text())
    assert len(entries) == 40 and seg.phonemization_failures == 1
    assert all(e["phonemes"] != e["text"] for e in entries)
    assert all(e["phonemes"].startswith("p:") for e in entries), "raw text is not phonemes"
    assert not any("BOOM" in e["text"] for e in entries)


def test_too_many_phonemiser_failures_abort_the_dataset(training_service, monkeypatch):
    svc = training_service()
    fail_on(svc, "BOOM")
    texts = [f"satz {i}" for i in range(7)] + [f"BOOM {i}" for i in range(3)]
    data = svc.root / "data" / "voice"
    data.mkdir(parents=True)
    seg = segmenter(monkeypatch)

    from phonemization import PhonemizationError

    with pytest.raises(PhonemizationError, match="3 of 10"):
        seg.generate_training_metadata(make_segments(svc.root, texts), data, "voice", "de")
    assert not (data / "train.json").exists(), "a half-built dataset was written"


def test_the_failure_threshold_is_configurable(training_service, monkeypatch):
    svc = training_service({"PHONEMIZER_MAX_FAILURE_FRACTION": "0.5"})
    fail_on(svc, "BOOM")
    texts = [f"satz {i}" for i in range(7)] + [f"BOOM {i}" for i in range(3)]
    data = svc.root / "data" / "voice"
    data.mkdir(parents=True)
    seg = segmenter(monkeypatch)
    seg.generate_training_metadata(make_segments(svc.root, texts), data, "voice", "de")
    assert seg.phonemization_failures == 3


def test_a_missing_phonemiser_is_an_error_not_raw_text(training_service, monkeypatch):
    svc = training_service()
    seg = segmenter(monkeypatch)
    data = svc.root / "data" / "voice"
    data.mkdir(parents=True)
    monkeypatch.setitem(sys.modules, "phonemizer", None)  # import raises ImportError

    from phonemization import PhonemizationError

    with pytest.raises(PhonemizationError, match="not installed"):
        seg.generate_training_metadata(make_segments(svc.root, ["ein satz", "noch ein satz"]), data, "voice", "de")
    assert not (data / "train.json").exists()


def test_prepare_dataset_reports_a_phonemiser_that_is_down(training_service):
    svc = training_service()
    files = clips(svc.root)

    def broken(*args, **kwargs):
        raise RuntimeError("libespeak-ng.so.1: cannot open shared object file")

    svc.phonemizer.phonemize = broken

    async def run():
        async with asgi_client(svc.module) as client:
            response = await prepare(client, segments=[segment(f) for f in files])
            assert response.status_code == 500
            assert "espeak" in response.json()["detail"]
            assert not (svc.root / "data" / "voice" / "train.json").exists()

    asyncio.run(run())


# --- /prepare-dataset details -----------------------------------------------------------------------------------


def test_the_response_counts_the_samples_that_were_written(training_service):
    svc = training_service()
    files = clips(svc.root, 4)

    async def run():
        async with asgi_client(svc.module) as client:
            response = await prepare(client, segments=[
                segment(files[0]), segment(files[1]), segment(files[2]),
                segment(files[3], start=1.0, end=0.5),  # an impossible range
            ])
            body = response.json()
            assert response.status_code == 200, response.text
            assert body["num_samples"] == 3 and body["num_skipped"] == 1
            assert body["num_segments_received"] == 4

    asyncio.run(run())


def test_only_the_requested_slice_of_each_recording_is_decoded(training_service, monkeypatch):
    """Loading the whole recording once per segment is quadratic in the number of segments."""
    svc = training_service()
    files = clips(svc.root, 2)
    spy = Spy(svc, monkeypatch)

    async def run():
        async with asgi_client(svc.module) as client:
            response = await prepare(client, segments=[
                segment(files[0], start=0.25, end=1.25), segment(files[1], start=0.5, end=1.5)])
            assert response.status_code == 200, response.text
            kwargs = [kw for _, kw in spy.opened]
            assert [round(k.get("offset", -1), 2) for k in kwargs] == [0.25, 0.5]
            assert [round(k.get("duration", -1), 2) for k in kwargs] == [1.0, 1.0]

    asyncio.run(run())


def test_over_long_or_empty_input_is_refused(training_service):
    svc = training_service({"MAX_TEXT_CHARS": "50"})
    files = clips(svc.root, 2)

    async def run():
        async with asgi_client(svc.module) as client:
            too_long = await prepare(client, segments=[segment(files[0], text="x" * 51), segment(files[1])])
            assert too_long.status_code == 400 and "MAX_TEXT_CHARS" in too_long.json()["detail"]
            assert (await prepare(client, segments=[])).status_code == 400
            fits = await prepare(client, segments=[segment(f, text="x" * 50) for f in files])
            assert fits.status_code == 200, fits.text

    asyncio.run(run())


# --- STT: confidence, language, errors --------------------------------------------------------------------------------


def stt_processor_module():
    import stt_processor  # the service directory is on sys.path while a training_service is loaded

    return stt_processor


@pytest.mark.parametrize("segment_data,expected", [
    ({"avg_logprob": -0.1}, pytest.approx(0.905, abs=0.001)),
    ({"avg_logprob": -1.0}, pytest.approx(0.368, abs=0.001)),
    ({"avg_logprob": 0.0}, 1.0),
    ({"avg_logprob": None}, 0.0),          # -inf/NaN, nulled by stt-service's JSON layer
    ({"avg_logprob": float("-inf")}, 0.0),
    ({"confidence": 0.7, "avg_logprob": -3.0}, 0.7),
    ({"confidence": 7}, 1.0),
    ({"text": "no score at all"}, None),
])
def test_segment_confidence_is_derived_from_avg_logprob(training_service, segment_data, expected):
    training_service()
    assert stt_processor_module().confidence_from_segment(segment_data) == expected


def test_low_confidence_segments_are_now_filtered(training_service, tmp_path):
    """stt-service returns avg_logprob only, so segment.get('confidence', 1.0) scored everything 1.0."""
    svc = training_service()
    module = stt_processor_module()
    audio = write_wav(tmp_path / "a.wav", seconds=6.0)
    processor = module.STTProcessor("http://stt")

    async def fake(self, path, return_segments=True, max_retries=3):
        return {"text": "x", "segments": [
            {"start": 0.0, "end": 2.0, "text": "sicherer satz hier", "avg_logprob": -0.1},
            {"start": 2.0, "end": 4.0, "text": "unsicherer satz da", "avg_logprob": -2.0},
        ]}

    processor.transcribe_audio_file = fake.__get__(processor)
    segments = asyncio.run(processor.process_audio_file(audio))
    assert [round(s.confidence, 2) for s in segments] == [0.9, 0.14]
    kept = processor.filter_segments_by_quality(segments, min_confidence=0.6, min_text_length=5)
    assert [s.text for s in kept] == ["sicherer satz hier"]


def test_stt_min_confidence_is_read_from_the_environment(training_service):
    for value, expected in (("0.85", 0.85), ("0", 0.0), ("", 0.6), ("banana", 0.6), ("1.5", 0.6)):
        svc = training_service({"STT_MIN_CONFIDENCE": value} if value else None)
        assert stt_processor_module().default_min_confidence() == expected, value


class FakeSession:
    """Answers session.post() from a script of (status, json-or-exception)."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.forms: list = []

    def post(self, url, data=None, timeout=None):
        self.calls += 1
        self.forms.append({name: value for name, value, _ in data.fields})
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        status, payload = step
        session = self

        class Response:
            def __init__(self):
                self.status = status

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def text(self):
                return json.dumps(payload)

            async def json(self):
                return payload

        return Response()


@pytest.fixture
def no_backoff(monkeypatch):
    """The retry loop sleeps 2 s and 4 s between attempts; tests do not wait."""
    real_sleep = asyncio.sleep

    async def quick(delay, *args, **kwargs):
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", quick)


def test_the_dataset_language_is_sent_to_the_stt_service(training_service, tmp_path):
    training_service()
    module = stt_processor_module()
    audio = write_wav(tmp_path / "a.wav")

    async def run(language):
        processor = module.STTProcessor("http://stt", language=language)
        processor.session = FakeSession([(200, {"text": "hallo", "segments": []})])
        await processor.transcribe_audio_file(audio)
        return processor.session.forms[0]["language"]

    assert asyncio.run(run("de")) == "de"
    assert asyncio.run(run(None)) == "auto"


def test_a_busy_stt_service_is_retried_and_a_rejection_is_not(training_service, tmp_path, no_backoff):
    training_service()
    module = stt_processor_module()
    audio = write_wav(tmp_path / "a.wav")

    async def run(script):
        processor = module.STTProcessor("http://stt")
        processor.session = FakeSession(script)
        try:
            return await processor.transcribe_audio_file(audio), processor.session.calls
        except module.STTError as exc:
            return exc, processor.session.calls

    result, calls = asyncio.run(run([(503, {"detail": "loading"}), (503, {"detail": "loading"}),
                                     (200, {"text": "hallo", "segments": []})]))
    assert result["text"] == "hallo" and calls == 3

    result, calls = asyncio.run(run([(400, {"detail": "Unsupported language"})]))
    assert isinstance(result, module.STTError) and "400" in str(result) and "Unsupported language" in str(result)
    assert calls == 1, "a 400 will not get better by asking again"

    result, calls = asyncio.run(run([(503, {})] * 3))
    assert isinstance(result, module.STTError) and "503" in str(result) and calls == 3


def test_a_request_that_times_out_is_not_started_again(training_service, tmp_path, no_backoff):
    """The timeout is an hour: three attempts would be three hours."""
    training_service()
    module = stt_processor_module()
    audio = write_wav(tmp_path / "a.wav")

    async def run():
        processor = module.STTProcessor("http://stt")
        processor.session = FakeSession([asyncio.TimeoutError()] * 3)
        with pytest.raises(module.STTError, match="timed out"):
            await processor.transcribe_audio_file(audio)
        return processor.session.calls

    assert asyncio.run(run()) == 1


def test_an_unreachable_stt_service_raises_instead_of_returning_no_segments(training_service, tmp_path, no_backoff):
    svc = training_service()
    module = stt_processor_module()
    audio = write_wav(tmp_path / "a.wav")
    refused = svc.aiohttp.ClientConnectorError("Connect call failed ('10.0.0.5', 8000)")

    async def run():
        processor = module.STTProcessor("http://stt")
        processor.session = FakeSession([refused] * 3)
        return await processor.process_audio_file(audio)

    with pytest.raises(module.STTError, match=r"3 attempts.*10\.0\.0\.5") as caught:
        asyncio.run(run())
    assert "10.0.0.5" in str(caught.value) and "3 attempts" in str(caught.value)


# --- the /train pipeline end to end --------------------------------------------------------------------------------------


def fake_segmenter_cuts(monkeypatch):
    """No ffmpeg in CI: cut the segments by writing a WAV of the right length."""
    import audio_segmenter

    async def extract(self, input_path, output_path, start_time, end_time, sample_rate=22050, channels=1):
        write_wav(output_path, seconds=end_time - start_time)
        return True

    monkeypatch.setattr(audio_segmenter.AudioSegmenter, "_find_ffmpeg", lambda self: "ffmpeg")
    monkeypatch.setattr(audio_segmenter.AudioSegmenter, "extract_audio_segment", extract)


def recording(tmp: Path, name="rec.wav", seconds=12.0) -> tuple:
    return ("audio_files", (name, write_wav(tmp / name, seconds=seconds).read_bytes(), "audio/wav"))


def test_train_runs_upload_to_export_in_the_dataset_language(training_service, monkeypatch, tmp_path):
    svc = training_service()
    fake_segmenter_cuts(monkeypatch)

    async def respond(processor, path):
        return stt_result(count=5)

    stt = install_fake_stt(monkeypatch, respond)
    stale_mel = svc.root / "data" / "voice" / "mel" / "voice_rec_0000.npy"
    stale_mel.parent.mkdir(parents=True)
    import numpy as np
    np.save(stale_mel, np.zeros((80, 1), dtype=np.float32))  # left by an earlier upload of different audio

    async def run():
        async with asgi_client(svc.module) as client:
            response = await client.post(
                "/train", data={"model_name": "voice", "language": "de", "epochs": 5},
                files=[recording(tmp_path)])
            assert response.status_code == 202, response.text
            job_id = response.json()["job_id"]
            await settle(svc)
            job = svc.module.training_jobs[job_id]
            assert job.status == "completed", job.message

            assert stt.instances[0].language == "de", "the STT call was not told the dataset language"
            assert {c.language for c in svc.phonemizer.calls} == {"de"}
            assert svc.pipeline.calls[0].request.language == "de"

            data = svc.root / "data" / "voice"
            train = json.loads((data / "train.json").read_text())
            val = json.loads((data / "val.json").read_text())
            assert len(train) + len(val) == 5
            assert all(e["phonemes"].startswith("p:") for e in train + val)
            for entry in train + val:
                assert (data / entry["mel_path"]).exists(), f"missing {entry['mel_path']}"
            assert np.load(stale_mel).shape == (80, 4), "a stale mel from other audio was kept"
            assert not (data / "temp_uploads").exists()

    asyncio.run(run())


@pytest.mark.parametrize("threshold,kept", [("0.6", 3), ("0.95", 0), ("0", 5)])
def test_the_confidence_threshold_decides_which_transcripts_train(training_service, monkeypatch, tmp_path,
                                                                 threshold, kept):
    svc = training_service({"STT_MIN_CONFIDENCE": threshold})
    fake_segmenter_cuts(monkeypatch)

    async def respond(processor, path):
        good = stt_result(text="ein sicher erkannter satz", count=3, avg_logprob=-0.1)
        bad = stt_result(text="ein unsicher erkannter satz", count=2, avg_logprob=-2.0)
        for i, s in enumerate(bad["segments"]):
            s["start"], s["end"] = 6.0 + i * 2, 8.0 + i * 2
        good["segments"] += bad["segments"]
        return good

    install_fake_stt(monkeypatch, respond)

    async def run():
        async with asgi_client(svc.module) as client:
            response = await client.post(
                "/train", data={"model_name": "voice", "epochs": 5}, files=[recording(tmp_path)])
            assert response.status_code == 202
            job_id = response.json()["job_id"]
            await settle(svc)
            job = svc.module.training_jobs[job_id]
            if kept == 0:
                assert job.status == "failed" and "confidence" in job.message
                return
            assert job.status == "completed", job.message
            entries = json.loads((svc.root / "data" / "voice" / "metadata.json").read_text())
            assert len(entries) == kept

    asyncio.run(run())


def test_an_stt_outage_fails_the_job_with_its_cause(training_service, monkeypatch, tmp_path, no_backoff):
    """It used to end as 'No valid training segments were created', cause only in a log."""
    svc = training_service()
    fake_segmenter_cuts(monkeypatch)

    async def run():
        async with asgi_client(svc.module) as client:
            response = await client.post(
                "/train", data={"model_name": "voice", "epochs": 5}, files=[recording(tmp_path)])
            job_id = response.json()["job_id"]
            await settle(svc)
            job = svc.module.training_jobs[job_id]
            assert job.status == "failed"
            assert "STT" in job.message and "unavailable" in job.message, job.message
            assert svc.pipeline.calls == []
            assert not (svc.root / "data" / "voice" / "temp_uploads").exists(), "uploads left behind"

    asyncio.run(run())


def test_uploads_with_the_same_name_do_not_overwrite_each_other(training_service, monkeypatch, tmp_path):
    svc = training_service()
    fake_segmenter_cuts(monkeypatch)
    seen: list = []

    async def respond(processor, path):
        seen.append(Path(path))
        return stt_result(count=3)

    install_fake_stt(monkeypatch, respond)

    async def run():
        async with asgi_client(svc.module) as client:
            response = await client.post(
                "/train", data={"model_name": "voice", "epochs": 5},
                files=[recording(tmp_path, "take.wav"), recording(tmp_path, "take.wav"),
                       recording(tmp_path, "take.mp3"), recording(tmp_path, "sub/../take.wav")])
            assert response.status_code == 202, response.text
            assert response.json()["files_uploaded"] == 4
            await settle(svc)
            stems = [p.stem.lower() for p in seen]
            assert len(stems) == 4 and len(set(stems)) == 4, (
                f"uploads share segment names and would overwrite each other: {stems}")

    asyncio.run(run())


def test_an_oversized_upload_is_refused_before_it_is_stored(training_service, tmp_path):
    svc = training_service({"MAX_UPLOAD_MB": "1"})

    async def run():
        async with asgi_client(svc.module) as client:
            big = ("audio_files", ("big.wav", b"\0" * (3 * 1024 * 1024), "audio/wav"))
            response = await client.post("/train", data={"model_name": "voice"}, files=[big])
            assert response.status_code == 413
            assert svc.module.training_jobs == {}
            assert not (svc.root / "data" / "voice").exists()
            assert len(svc.module.active_runs) == 0

    asyncio.run(run())


def test_a_chunked_upload_is_cut_off_at_the_limit_not_read_to_the_end(training_service):
    svc = training_service({"MAX_UPLOAD_MB": "1"})
    pulled = 0
    head = (b"--b\r\nContent-Disposition: form-data; name=\"model_name\"\r\n\r\nvoice\r\n"
            b"--b\r\nContent-Disposition: form-data; name=\"audio_files\"; filename=\"x.wav\"\r\n"
            b"Content-Type: audio/wav\r\n\r\n")

    async def body():
        nonlocal pulled
        yield head
        for _ in range(400):                       # 100 MiB on offer
            pulled += 1
            yield b"\0" * (256 * 1024)

    async def run():
        async with asgi_client(svc.module) as client:
            response = await client.post(
                "/train", content=body(), headers={"content-type": "multipart/form-data; boundary=b"})
            assert response.status_code == 413
            assert pulled < 20, f"the server kept reading after the cap: {pulled} chunks consumed"
            assert svc.module.training_jobs == {}

    asyncio.run(run())


def test_an_upload_within_the_limit_that_has_no_audio_is_refused(training_service):
    svc = training_service()

    async def run():
        async with asgi_client(svc.module) as client:
            response = await client.post(
                "/train", data={"model_name": "voice"},
                files=[("audio_files", ("", b"", "audio/wav"))])
            assert response.status_code in (400, 422), response.text
            assert svc.module.training_jobs == {}
            assert len(svc.module.active_runs) == 0

    asyncio.run(run())


@pytest.mark.parametrize("field,value", [
    ("sample_rate", "16000"), ("language", "klingon"), ("batch_size", "0"), ("epochs", "0"),
])
def test_train_validates_what_it_would_otherwise_train_on(training_service, tmp_path, field, value):
    svc = training_service()

    async def run():
        async with asgi_client(svc.module) as client:
            response = await client.post(
                "/train", data={"model_name": "voice", field: value}, files=[recording(tmp_path, seconds=1.0)])
            assert response.status_code == 400, response.text
            assert svc.module.training_jobs == {}
            assert len(svc.module.active_runs) == 0

    asyncio.run(run())


@pytest.mark.parametrize("path,field,value", [
    (path, field, value)
    for path, field, values in (
        ("/train-from-dataset", "epochs", ("0", "-5", "1000000000")),
        ("/retrain-from-segments", "epochs", ("0", "-5", "1000000000")),
        # extra_epochs=0 is meaningful here: "continue to the original total".
        ("/resume-training", "extra_epochs", ("-5", "1000000000")),
    )
    for value in values
])
def test_every_other_way_of_starting_a_job_bounds_the_epoch_count_too(training_service, path, field, value):
    """`epochs=0` produced a job reporting 0/0 that divided by zero in the resume path's own
    progress calculation; the bound has to hold at every entry point, not only /train."""
    svc = training_service()
    make_dataset(svc.root, "voice")
    write_wav(svc.root / "data" / "voice" / "audio" / "voice_clip_0.wav")
    write_job_state(svc.root, "job-a", "voice", status="training")

    async def run():
        async with asgi_client(svc.module) as client:
            response = await client.post(path, data={"model_name": "voice", field: value})
            assert response.status_code == 400, response.text
            assert field in response.text
            assert svc.module.training_jobs == {}
            assert len(svc.module.active_runs) == 0
            assert svc.pipeline.calls == []

    asyncio.run(run())


def _segmenter(training_service, monkeypatch, fake_run):
    """An AudioSegmenter whose ffmpeg call goes to *fake_run* (the service is put on sys.path first)."""
    training_service()
    import audio_segmenter

    monkeypatch.setattr(audio_segmenter.AudioSegmenter, "_find_ffmpeg", lambda self: "ffmpeg")
    monkeypatch.setattr(audio_segmenter.subprocess, "run", fake_run)
    return audio_segmenter.AudioSegmenter()


def test_a_segment_is_cut_by_seeking_the_input_off_the_event_loop_and_under_a_timeout(
        training_service, tmp_path, monkeypatch):
    """`-ss` after `-i` is output seeking: ffmpeg decodes the file from the start for every cut,
    so N segments cost N full decodes. The call must also leave the event loop (it runs in a
    BackgroundTask that /health shares) and be bounded (a wedged ffmpeg hung it for good)."""
    seen = {}

    def fake_run(cmd, **kwargs):
        seen.update(cmd=list(cmd), kwargs=kwargs, thread=threading.get_ident())
        Path(cmd[-1]).write_bytes(b"RIFF" + b"\0" * 40)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    segmenter = _segmenter(training_service, monkeypatch, fake_run)

    async def cut():
        return threading.get_ident(), await segmenter.extract_audio_segment(
            tmp_path / "in.wav", tmp_path / "out" / "seg.wav", 12.5, 15.0)

    loop_thread, ok = asyncio.run(cut())

    cmd = seen["cmd"]
    assert ok is True
    assert cmd.index("-ss") < cmd.index("-i"), f"-ss after -i decodes the whole file for every cut: {cmd}"
    assert cmd[cmd.index("-ss") + 1] == "12.5" and cmd[cmd.index("-t") + 1] == "2.5"
    assert cmd[cmd.index("-i") + 1] == str(tmp_path / "in.wav")
    assert seen["kwargs"]["timeout"] > 0, "ffmpeg runs with no timeout"
    assert seen["thread"] != loop_thread, "ffmpeg was run on the event loop thread"


def test_a_wedged_ffmpeg_is_a_failed_cut_not_a_hung_job(training_service, tmp_path, monkeypatch):
    def hung(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    segmenter = _segmenter(training_service, monkeypatch, hung)

    ok = asyncio.run(segmenter.extract_audio_segment(tmp_path / "in.wav", tmp_path / "seg.wav", 0.0, 2.0))

    assert ok is False


def test_retrain_from_segments_reports_an_stt_outage_and_uses_the_language(training_service, monkeypatch):
    svc = training_service()
    for index in range(3):
        write_wav(svc.root / "data" / "voice" / "audio" / f"voice_clip_{index}.wav")

    async def down(processor, path):
        raise sys.modules["stt_processor"].STTError("STT service at http://stt unavailable after 3 attempts")

    stt = install_fake_stt(monkeypatch, down)

    async def run():
        async with asgi_client(svc.module) as client:
            response = await client.post("/retrain-from-segments", data={"model_name": "voice", "language": "de"})
            job_id = response.json()["job_id"]
            await settle(svc)
            job = svc.module.training_jobs[job_id]
            assert job.status == "failed"
            assert "unavailable" in job.message and "3 of 3" in job.message, job.message
            assert stt.instances[0].language == "de"

    asyncio.run(run())


def test_retrain_from_segments_applies_the_confidence_threshold(training_service, monkeypatch):
    svc = training_service()
    for index in range(4):
        write_wav(svc.root / "data" / "voice" / "audio" / f"voice_clip_{index}.wav")

    async def respond(processor, path):
        unsure = path.stem.endswith("3")
        return stt_result(text="ein ganzer satz zum lernen", seconds=2.0, avg_logprob=-2.5 if unsure else -0.1)

    install_fake_stt(monkeypatch, respond)
    fake_segmenter_cuts(monkeypatch)  # AudioSegmenter() looks for ffmpeg even to only write metadata

    async def run():
        async with asgi_client(svc.module) as client:
            response = await client.post("/retrain-from-segments", data={"model_name": "voice"})
            job_id = response.json()["job_id"]
            await settle(svc)
            assert status_of(svc, job_id) == "completed", svc.module.training_jobs[job_id].message
            entries = json.loads((svc.root / "data" / "voice" / "metadata.json").read_text())
            assert len(entries) == 3 and not any(e["audio_path"].endswith("clip_3.wav") for e in entries)

    asyncio.run(run())


# --- the rules themselves ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("value,code,voice,stt", [
    ("de", "de", "de", "de"),
    (" DE ", "de", "de", "de"),
    ("de_DE", "de", "de", "de"),
    ("de-AT", "de", "de", "de"),
    ("en", "en", "en-us", "en"),
    ("en-GB", "en-gb", "en-gb", "en"),
    ("pt-BR", "pt-br", "pt-br", "pt"),
    ("fr", "fr", "fr-fr", "fr"),
])
def test_language_codes_map_to_an_espeak_voice_and_an_stt_code(training_service, value, code, voice, stt):
    training_service()
    import phonemization

    assert phonemization.normalize_language(value) == code
    assert phonemization.espeak_voice(value) == voice
    assert phonemization.stt_language(value) == stt


@pytest.mark.parametrize("value", ["", None, "xx", "klingon", "zh"])
def test_unsupported_languages_raise_with_the_supported_list(training_service, value):
    training_service()
    import phonemization

    with pytest.raises(ValueError, match="Supported: de, en, es, fr, it, nl, pt, ru"):
        phonemization.normalize_language(value)


def test_stt_language_is_auto_only_when_no_language_is_known(training_service):
    training_service()
    import phonemization

    assert phonemization.stt_language(None) == "auto"
    assert phonemization.stt_language("") == "auto"


@pytest.mark.parametrize("value", ["", "   ", "a\x00b.wav", "gopher://host/a.wav", "s3://bucket/a.wav"])
def test_malformed_or_exotic_audio_paths_are_refused(training_service, value):
    training_service()
    import audio_sources

    with pytest.raises(audio_sources.AudioSourceError):
        audio_sources.resolve_audio_source(value)


def test_a_file_inside_data_resolves_to_its_real_location(training_service):
    svc = training_service()
    import audio_sources

    kind, resolved = audio_sources.resolve_audio_source("data/voice/raw/a.wav")
    assert kind == "file" and resolved == (svc.root / "data" / "voice" / "raw" / "a.wav").resolve()


def test_cancelling_an_upload_job_stops_the_remaining_stt_work(training_service, monkeypatch, tmp_path):
    """Each file can be many minutes of STT; a cancelled job should not wait for the rest."""
    svc = training_service()
    fake_segmenter_cuts(monkeypatch)
    transcribed: list = []

    async def respond(processor, path):
        transcribed.append(Path(path).name)
        return stt_result(count=3)

    async def run():
        gate = asyncio.Event()
        stt = install_fake_stt(monkeypatch, respond, gate)
        async with asgi_client(svc.module) as client:
            post = asyncio.create_task(client.post(
                "/train", data={"model_name": "voice", "epochs": 5},
                files=[recording(tmp_path, "one.wav"), recording(tmp_path, "two.wav"),
                       recording(tmp_path, "three.wav")]))
            try:
                await asyncio.wait_for(_until(lambda: stt.waiting >= 1), 5)
                job_id = next(iter(svc.module.training_jobs))
                assert (await client.delete(f"/job/{job_id}")).status_code == 200
            finally:
                gate.set()
            await settle(svc)
            await asyncio.wait_for(post, 5)
            assert transcribed == ["one.wav"], f"the cancelled job went on to {transcribed}"
            assert status_of(svc, job_id) == "cancelled"
            assert svc.pipeline.calls == [] and svc.exporter.calls == []
            assert not (svc.root / "data" / "voice" / "temp_uploads").exists()

    asyncio.run(run())


async def _until(predicate):
    while not predicate():
        await asyncio.sleep(0.005)


def test_a_recording_longer_than_one_chunk_is_transcribed_once_not_for_ever(training_service, tmp_path):
    """The chunk loop stepped back by the overlap from the end of the file and never left."""
    training_service()
    module = stt_processor_module()
    audio = write_wav(tmp_path / "long.wav", seconds=130.0, rate=8000)
    chunks: list = []

    async def run():
        processor = module.STTProcessor("http://stt")

        async def one_chunk(chunk_path):
            chunks.append(chunk_path.name)
            assert len(chunks) < 10, "the same tail is being transcribed over and over"
            return []

        processor.process_audio_file = one_chunk
        return await asyncio.wait_for(processor.process_large_audio_file(audio, max_chunk_duration=60.0), 10)

    assert asyncio.run(run()) == []
    assert chunks == ["chunk_000.wav", "chunk_001.wav", "chunk_002.wav"]
