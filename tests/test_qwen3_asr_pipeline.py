"""Chunking, decoding, joining and upload spooling of the Qwen3-ASR service.

These drive ``audio_pipeline`` directly (numpy and the stdlib are enough); the
endpoint behaviour built on top of it is in ``test_qwen3_asr_endpoints``.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile

from test_qwen3_asr_harness import (
    RATE, FakeFfmpeg, encoded_wav, load_audio_pipeline, process_alive, run, silent_wav,
    speech_levels, tmp_files, use_tmpdir, wav_from_int16,
)

pipeline, fake_librosa = load_audio_pipeline()


def seconds_of(n: float) -> int:
    return int(n * RATE)


def gapped_noise(seconds: float, *, gap_every: float = 3.0, gap_s: float = 0.3, seed: int = 0) -> np.ndarray:
    """Loud noise with a silent gap every *gap_every* seconds."""
    rng = np.random.default_rng(seed)
    samples = rng.uniform(-0.5, 0.5, seconds_of(seconds)).astype(np.float32)
    step = seconds_of(gap_every)
    for start in range(step, len(samples), step):
        samples[start:start + seconds_of(gap_s)] = 0.0
    return samples


# --- plan_chunks -------------------------------------------------------------------

def assert_tiles(chunks, total, *, chunk_s):
    """The pieces cover the audio exactly once and none is longer than *chunk_s*."""
    assert chunks[0].start == 0
    assert chunks[-1].end == total
    for previous, current in zip(chunks, chunks[1:]):
        assert current.start == previous.end
    for chunk in chunks:
        assert 0 < chunk.end - chunk.start <= int(chunk_s * RATE), chunk
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_audio_up_to_the_chunk_length_is_one_piece():
    for seconds in (0.2, 5.0, 30.0):
        samples = np.ones(seconds_of(seconds), dtype=np.float32)
        chunks = pipeline.plan_chunks(samples, chunk_s=30.0)
        assert [(c.start, c.end) for c in chunks] == [(0, len(samples))]


def test_no_audio_is_no_pieces():
    assert pipeline.plan_chunks(np.zeros(0, dtype=np.float32), chunk_s=30.0) == []


def test_long_audio_is_cut_in_pauses_not_through_speech():
    samples = gapped_noise(100.0)
    chunks = pipeline.plan_chunks(samples, chunk_s=30.0)
    assert len(chunks) >= 4
    assert_tiles(chunks, len(samples), chunk_s=30.0)
    for chunk in chunks[:-1]:
        around = samples[chunk.end - 400:chunk.end + 400]
        assert not around.any(), f"cut at {chunk.end_s:.2f}s is not inside a silent gap"


def test_a_cut_prefers_the_longest_piece_among_equally_quiet_places():
    samples = np.zeros(seconds_of(100.0), dtype=np.float32)  # digital silence everywhere
    chunks = pipeline.plan_chunks(samples, chunk_s=30.0)
    assert_tiles(chunks, len(samples), chunk_s=30.0)
    # Ties resolve to the latest window, so pieces are nearly full length, not slivers.
    assert all(c.end - c.start > seconds_of(24.0) for c in chunks[:-1])


def test_audio_without_any_pause_is_still_cut_within_the_limit():
    samples = np.full(seconds_of(95.0), 0.3, dtype=np.float32)
    assert_tiles(pipeline.plan_chunks(samples, chunk_s=30.0), len(samples), chunk_s=30.0)


@pytest.mark.parametrize("seconds", [30.01, 30.4, 31.0, 59.9, 60.0, 60.5, 61.0, 89.99, 120.3])
def test_the_last_piece_is_never_a_sliver(seconds):
    samples = gapped_noise(seconds, seed=int(seconds * 100))
    chunks = pipeline.plan_chunks(samples, chunk_s=30.0)
    assert_tiles(chunks, len(samples), chunk_s=30.0)
    assert len(chunks) >= 2
    assert chunks[-1].end - chunks[-1].start >= seconds_of(pipeline.MIN_TAIL_S)


def test_a_short_chunk_length_works():
    samples = gapped_noise(12.0, gap_every=1.0)
    chunks = pipeline.plan_chunks(samples, chunk_s=5.0)
    assert_tiles(chunks, len(samples), chunk_s=5.0)


def test_chunk_times_are_in_seconds():
    chunks = pipeline.plan_chunks(gapped_noise(70.0), chunk_s=30.0)
    assert chunks[0].start_s == 0.0
    assert chunks[0].end_s == chunks[0].end / RATE
    assert chunks[1].start_s == chunks[0].end_s


def test_a_clip_shorter_than_the_encoder_minimum_is_padded_with_silence():
    short = np.full(seconds_of(0.2), 0.5, dtype=np.float32)
    padded = pipeline.model_input(short)
    assert len(padded) == seconds_of(pipeline.MIN_MODEL_INPUT_S)
    assert padded.dtype == np.float32
    assert (padded[: len(short)] == short).all() and not padded[len(short):].any()
    longer = np.ones(seconds_of(1.0), dtype=np.float32)
    assert pipeline.model_input(longer) is longer


# --- join_texts --------------------------------------------------------------------

@pytest.mark.parametrize("parts,expected", [
    (["Hallo Welt", "wie geht es"], "Hallo Welt wie geht es"),
    (["Satz eins.", "Satz zwei."], "Satz eins. Satz zwei."),
    (["", "  ", "nur das"], "nur das"),
    ([" vorn ", " hinten "], "vorn hinten"),
    (["你好", "世界"], "你好世界"),
    (["Hello", "世界"], "Hello世界"),
    (["こんにちは", "world"], "こんにちはworld"),
    (["สวัสดี", "ครับ"], "สวัสดีครับ"),
    ([], ""),
])
def test_pieces_are_joined_the_way_the_language_is_written(parts, expected):
    assert pipeline.join_texts(parts) == expected


# --- env_number --------------------------------------------------------------------

def test_a_bad_tuning_knob_falls_back_instead_of_stopping_the_service(monkeypatch):
    monkeypatch.setenv("KNOB", "abc")
    assert pipeline.env_number("KNOB", 30.0) == 30.0
    monkeypatch.setenv("KNOB", "2")
    assert pipeline.env_number("KNOB", 30.0, minimum=5.0) == 30.0
    monkeypatch.setenv("KNOB", "9999")
    assert pipeline.env_number("KNOB", 30.0, maximum=600.0) == 600.0
    monkeypatch.setenv("KNOB", " 12 ")
    assert pipeline.env_number("KNOB", 30.0) == 12.0
    monkeypatch.setenv("KNOB", "")
    assert pipeline.env_number("KNOB", 30.0) == 30.0
    monkeypatch.delenv("KNOB")
    assert pipeline.env_number("KNOB", 8, cast=int) == 8


# --- duration ------------------------------------------------------------------------

def test_duration_limit_allows_the_limit_itself_and_zero_disables_it():
    pipeline.check_duration(10.0, 10.0)
    pipeline.check_duration(10.4, 10.0)
    pipeline.check_duration(None, 10.0)
    pipeline.check_duration(99999.0, 0.0)
    with pytest.raises(HTTPException) as info:
        pipeline.check_duration(10.6, 10.0)
    assert info.value.status_code == 413
    assert "MAX_AUDIO_SECONDS" in info.value.detail


def test_probe_reads_the_header_and_falls_back_for_other_containers(tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(silent_wav(2.0))
    before = fake_librosa.duration_calls
    assert pipeline.probe_duration(str(wav)) == pytest.approx(2.0)
    assert fake_librosa.duration_calls == before  # libsndfile answered; no decode fallback
    junk = tmp_path / "junk.bin"
    junk.write_bytes(b"not audio at all")
    assert pipeline.probe_duration(str(junk)) is None


# --- spool_upload ------------------------------------------------------------------

class RecordingFile(io.BytesIO):
    """Records the size of every read, so a whole-file read shows up."""

    def __init__(self, data):
        super().__init__(data)
        self.read_sizes = []

    def read(self, size=-1):
        self.read_sizes.append(size)
        return super().read(size)


def upload_of(data: bytes, *, filename="clip.wav", size="auto", file=None) -> UploadFile:
    return UploadFile(
        file=file or io.BytesIO(data), filename=filename, size=len(data) if size == "auto" else size,
    )


def test_an_upload_is_copied_in_bounded_reads(tmp_path, monkeypatch):
    use_tmpdir(monkeypatch, tmp_path)
    data = b"x" * (3 * 1024 * 1024 + 17)
    source = RecordingFile(data)
    path, size = run(pipeline.spool_upload(upload_of(data, file=source), max_bytes=10 * 1024 * 1024))
    assert size == len(data)
    assert open(path, "rb").read() == data
    assert all(isinstance(n, int) and 0 < n <= pipeline.MIB for n in source.read_sizes), source.read_sizes


def test_the_temp_file_suffix_cannot_escape_the_temp_directory(tmp_path, monkeypatch):
    use_tmpdir(monkeypatch, tmp_path)
    path, _ = run(pipeline.spool_upload(upload_of(b"abc", filename="../../etc/x.m4a;rm -rf"), max_bytes=1024))
    assert path.startswith(str(tmp_path))
    assert "/" not in path[len(str(tmp_path)) + 1:]
    path2, _ = run(pipeline.spool_upload(upload_of(b"abc", filename=None), max_bytes=1024))
    assert path2.endswith(".wav")


def test_an_upload_over_the_limit_is_refused_when_its_size_is_declared(tmp_path, monkeypatch):
    use_tmpdir(monkeypatch, tmp_path)
    with pytest.raises(HTTPException) as info:
        run(pipeline.spool_upload(upload_of(b"x" * 2048), max_bytes=1024))
    assert info.value.status_code == 413 and "MAX_UPLOAD_MB" in info.value.detail
    assert tmp_files(tmp_path) == []


def test_an_upload_over_the_limit_is_refused_while_streaming_and_leaves_nothing(tmp_path, monkeypatch):
    use_tmpdir(monkeypatch, tmp_path)
    with pytest.raises(HTTPException) as info:
        run(pipeline.spool_upload(upload_of(b"x" * (2 * pipeline.MIB + 5), size=None), max_bytes=pipeline.MIB))
    assert info.value.status_code == 413
    assert tmp_files(tmp_path) == []


def test_an_upload_of_exactly_the_limit_is_accepted(tmp_path, monkeypatch):
    use_tmpdir(monkeypatch, tmp_path)
    path, size = run(pipeline.spool_upload(upload_of(b"x" * 1024), max_bytes=1024))
    assert size == 1024


def test_an_empty_upload_is_a_client_error_and_leaves_nothing(tmp_path, monkeypatch):
    use_tmpdir(monkeypatch, tmp_path)
    with pytest.raises(HTTPException) as info:
        run(pipeline.spool_upload(upload_of(b""), max_bytes=1024))
    assert info.value.status_code == 400
    assert tmp_files(tmp_path) == []


# --- decode_audio ------------------------------------------------------------------

@pytest.fixture
def ffmpeg(tmp_path, monkeypatch):
    fake = FakeFfmpeg(tmp_path)
    monkeypatch.setenv("FAKE_FFMPEG_DIR", str(fake.dir))
    return fake


def write_clip(tmp_path, data: bytes, name="clip.wav") -> str:
    path = tmp_path / name
    path.write_bytes(data)
    return str(path)


def test_ffmpeg_is_asked_for_16k_mono_float_on_a_pipe_and_bounded(tmp_path, ffmpeg):
    path = write_clip(tmp_path, encoded_wav(speech_levels(40)))
    samples = pipeline.decode_audio(path, cap_seconds=4.0, ffmpeg=str(ffmpeg.path))
    args = ffmpeg.calls()[0]["args"]
    assert args[args.index("-ar") + 1] == "16000"
    assert args[args.index("-ac") + 1] == "1"
    assert args[args.index("-t") + 1] == "4"
    assert args[args.index("-f") + 1] == "f32le" and args[-1] == "pipe:1"
    assert samples.dtype == np.float32
    assert len(samples) == seconds_of(4.0)


def test_ffmpeg_output_is_the_decoded_waveform(tmp_path, ffmpeg):
    pcm = (np.sin(np.arange(RATE) / 20.0) * 12000).astype(np.int16)
    path = write_clip(tmp_path, wav_from_int16(pcm))
    samples = pipeline.decode_audio(path, ffmpeg=str(ffmpeg.path))
    assert np.allclose(samples, pcm.astype(np.float32) / 32768.0)
    assert "-t" not in ffmpeg.calls()[0]["args"]  # no cap requested, none applied


def test_a_failing_ffmpeg_falls_back_to_the_python_decoder(tmp_path, ffmpeg):
    ffmpeg.mode = "fail"
    path = write_clip(tmp_path, silent_wav(2.0))
    before = len(fake_librosa.load_calls)
    samples = pipeline.decode_audio(path, cap_seconds=1.5, ffmpeg=str(ffmpeg.path))
    assert len(samples) == seconds_of(1.5)
    call = fake_librosa.load_calls[before]
    assert call["sr"] == RATE and call["mono"] is True and call["duration"] == 1.5


def test_a_hanging_ffmpeg_is_killed_and_the_python_decoder_takes_over(tmp_path, ffmpeg):
    ffmpeg.mode = "hang"
    path = write_clip(tmp_path, silent_wav(1.0))
    samples = pipeline.decode_audio(path, ffmpeg=str(ffmpeg.path), timeout=1.0)
    assert len(samples) == seconds_of(1.0)
    pid = ffmpeg.hung_pid()
    assert pid is not None and not process_alive(pid)


def test_audio_no_decoder_can_read_is_a_decode_error(tmp_path, ffmpeg):
    path = write_clip(tmp_path, b"this is not audio")
    with pytest.raises(pipeline.AudioDecodeError):
        pipeline.decode_audio(path, ffmpeg=str(ffmpeg.path))
    with pytest.raises(pipeline.AudioDecodeError):
        pipeline.decode_audio(path, ffmpeg=None)


def test_a_missing_ffmpeg_binary_falls_back(tmp_path):
    path = write_clip(tmp_path, silent_wav(1.0))
    samples = pipeline.decode_audio(path, ffmpeg=str(tmp_path / "no-such-ffmpeg"))
    assert len(samples) == seconds_of(1.0)
