"""Unit tests for the helpers of the transformers ASR services (no torch, no transformers).

``transformers_asr.py`` (shared by cohere-asr-service and granite-asr-service; the
copies are kept identical by test_transformers_asr_shared.py), ``cohere_support.py``
and ``granite_support.py``.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from transformers_asr_loader import RATE, load_module

ta = load_module("cohere", "transformers_asr")
cs = load_module("cohere", "cohere_support")
gs = load_module("granite", "granite_support")


def seconds(n: float) -> int:
    return int(round(n * RATE))


def tone(duration_s: float, amplitude: float = 0.3, freq: float = 220.0) -> np.ndarray:
    t = np.arange(seconds(duration_s)) / RATE
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# --- languages ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", [None, "", "  ", "auto", "AUTO", " Auto "])
def test_auto_and_no_language_mean_the_default(raw):
    assert ta.resolve_language(raw, default="de", supported={"de", "en"}, model="M") == "de"


@pytest.mark.parametrize("raw, expected", [("de", "de"), ("DE", "de"), ("de-DE", "de"), ("en_GB", "en"), (" en ", "en")])
def test_codes_and_locale_tags_reduce_to_the_language(raw, expected):
    assert ta.resolve_language(raw, default="de", supported={"de", "en"}, model="M") == expected


@pytest.mark.parametrize("raw", ["sv", "German", "xx-YY", "zz"])
def test_a_language_the_model_does_not_know_is_refused_with_the_list_and_the_default(raw):
    with pytest.raises(ta.UnsupportedLanguage) as caught:
        ta.resolve_language(raw, default="de", supported={"de", "en"}, model="Cohere Transcribe")
    message = str(caught.value)
    assert raw.strip() in message and "Cohere Transcribe" in message
    assert "Supported: de, en" in message and "('de')" in message


def test_an_overlong_language_is_cut_in_the_refusal():
    with pytest.raises(ta.UnsupportedLanguage) as caught:
        ta.resolve_language("x" * 500, default="de", supported={"de"}, model="M")
    assert len(str(caught.value)) < 250


@pytest.mark.parametrize("raw, expected", [("de", "de"), ("fr-FR", "fr"), ("", "de"), ("auto", "de"), (None, "de"), ("sv", "de")])
def test_the_default_language_setting_falls_back_when_the_model_lacks_it(raw, expected):
    assert ta.default_language_from_env(raw, supported={"de", "fr"}, fallback="de") == expected


# --- configuration ------------------------------------------------------------------------


@pytest.mark.parametrize("raw, expected", [("true", True), ("1", True), ("ON", True), ("false", False), ("0", False),
                                           ("", True), ("maybe", True)])
def test_env_flag(monkeypatch, raw, expected):
    monkeypatch.setenv("X_FLAG", raw)
    assert ta.env_flag("X_FLAG", True) is expected


@pytest.mark.parametrize("raw, expected", [("silero", "silero"), ("ENERGY", "energy"), ("off", "off"),
                                           ("", "silero"), ("webrtc", "silero")])
def test_env_choice(monkeypatch, raw, expected):
    monkeypatch.setenv("X_VAD", raw)
    assert ta.env_choice("X_VAD", "silero", ta.VAD_MODES) == expected


def test_a_mounted_checkpoint_path_is_not_shown_to_callers():
    assert ta.public_model_name("/models/private/cohere-ft") == "cohere-ft"
    assert ta.public_model_name("evewashere/cohere-transcribe-03-2026-ungated") == "evewashere/cohere-transcribe-03-2026-ungated"
    assert ta.is_local_model("/models/x") and ta.is_local_model("./x") and not ta.is_local_model("org/name")


@pytest.mark.parametrize("exc, expected", [
    (type("OutOfMemoryError", (RuntimeError,), {})("x"), True),
    (RuntimeError("CUDA out of memory. Tried to allocate"), True),
    (RuntimeError("shape mismatch"), False),
])
def test_out_of_memory_is_recognised(exc, expected):
    assert ta.is_cuda_oom(exc) is expected


def test_the_token_cap_is_generous_for_speech_and_has_a_floor():
    assert ta.max_new_tokens_for(30) == 600
    assert ta.max_new_tokens_for(5) == 256
    assert ta.max_new_tokens_for(12.3) == 256


def test_batches_split_in_order():
    assert ta.batches([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    assert ta.batches([1, 2], 0) == [[1], [2]], "a size below 1 counts as 1"
    assert ta.batches([], 4) == []


# --- weights ------------------------------------------------------------------------------


@pytest.mark.parametrize("setting, default, expected", [
    (None, "a" * 64, "a" * 64),
    ("", None, None),
    ("off", "a" * 64, None),
    ("SKIP", "a" * 64, None),
    ("B" * 64, None, "b" * 64),
])
def test_the_expected_digest_comes_from_the_setting_or_the_default(setting, default, expected):
    assert ta.expected_sha256(setting, default) == expected


@pytest.mark.parametrize("setting", ["abc", "z" * 64, "a" * 63, "sha256:" + "a" * 64])
def test_a_digest_setting_that_is_neither_a_digest_nor_off_is_an_error_not_a_silent_skip(setting):
    with pytest.raises(ValueError):
        ta.expected_sha256(setting, "a" * 64)


def _weights(tmp_path, data: bytes = b"weights"):
    (tmp_path / "model.safetensors").write_bytes(data)
    return str(tmp_path), hashlib.sha256(data).hexdigest()


def test_matching_weights_pass_and_the_digest_is_returned(tmp_path):
    directory, digest = _weights(tmp_path)
    assert ta.verify_weights(directory, digest, size=7) == digest


def test_no_expected_digest_reads_nothing(tmp_path):
    calls = []
    assert ta.verify_weights(str(tmp_path), None, hasher=lambda p: calls.append(p) or "") is None
    assert calls == []


def test_other_weights_are_refused(tmp_path):
    directory, _ = _weights(tmp_path)
    with pytest.raises(ta.WeightsMismatch, match="refusing to load"):
        ta.verify_weights(directory, "0" * 64)


def test_a_truncated_file_is_refused_by_its_size_before_it_is_hashed(tmp_path):
    directory, digest = _weights(tmp_path)
    calls = []
    with pytest.raises(ta.WeightsMismatch, match="bytes"):
        ta.verify_weights(directory, digest, size=4131862976, hasher=lambda p: calls.append(p) or digest)
    assert calls == []


def test_a_missing_weights_file_is_a_mismatch(tmp_path):
    with pytest.raises(ta.WeightsMismatch, match="missing"):
        ta.verify_weights(str(tmp_path), "0" * 64)


def test_weights_already_verified_are_not_read_again(tmp_path):
    directory, digest = _weights(tmp_path, b"hashed once")
    calls = []

    def hasher(path):
        calls.append(path)
        return digest

    ta.verify_weights(directory, digest, hasher=hasher)
    ta.verify_weights(directory, digest, hasher=hasher)
    assert len(calls) == 1, "a reload after an idle unload must not hash 4 GB again"


def test_changed_weights_are_hashed_again(tmp_path):
    directory, digest = _weights(tmp_path, b"first")
    ta.verify_weights(directory, digest)
    (tmp_path / "model.safetensors").write_bytes(b"second, longer")
    with pytest.raises(ta.WeightsMismatch):
        ta.verify_weights(directory, digest)


def test_the_download_never_fetches_remote_code_or_a_second_checkpoint():
    import fnmatch

    def wanted(name):
        return any(fnmatch.fnmatch(name, pattern) for pattern in ta.DOWNLOAD_PATTERNS)

    for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer.model", "merges.txt",
                 "chat_template.jinja", "model-00001-of-00002.safetensors", "preprocessor_config.json"):
        assert wanted(name), name
    for name in ("modeling_cohere_asr.py", "processing_cohere_asr.py", "out_llm.safetensors",
                 "demo/voxpopuli_test_en_demo.wav", "multilingual_sample.wav", "model.sig",
                 "assets/plot.png", "README.md"):
        assert not wanted(name), name


# --- the energy gate -----------------------------------------------------------------------


def test_digital_silence_has_no_speech():
    assert ta.energy_regions(np.zeros(seconds(5), dtype=np.float32)) == []


def test_steady_noise_at_any_level_is_gated_out_as_a_whole():
    rng = np.random.default_rng(0)
    for level in (0.003, 0.05, 0.3):
        noise = rng.normal(0, level, seconds(8)).astype(np.float32)
        assert ta.energy_regions(noise) == [], level


def test_speech_like_bursts_over_a_quiet_floor_are_found_with_padding():
    rng = np.random.default_rng(1)
    audio = rng.normal(0, 0.002, seconds(6)).astype(np.float32)
    audio[seconds(1):seconds(2)] += tone(1.0)
    audio[seconds(4):seconds(5)] += tone(1.0)
    regions = ta.energy_regions(audio)
    assert len(regions) == 2
    (a1, b1), (a2, b2) = regions
    assert a1 <= seconds(1) - seconds(0.2) and b1 >= seconds(2) + seconds(0.2)
    assert a2 <= seconds(4) - seconds(0.2) and b2 >= seconds(5) + seconds(0.2)


def test_a_click_shorter_than_the_minimum_run_is_not_speech():
    audio = np.zeros(seconds(3), dtype=np.float32)
    audio[seconds(1):seconds(1) + seconds(0.03)] = 0.5
    assert ta.energy_regions(audio) == []


# --- the speech gate ------------------------------------------------------------------------


def test_off_passes_the_whole_file():
    gate = ta.SpeechGate("off")
    assert gate.regions(np.ones(100, dtype=np.float32)) == [(0, 100)]
    assert gate.describe() == "off"


def test_an_empty_file_has_no_regions_in_any_mode():
    for mode in ta.VAD_MODES:
        assert ta.SpeechGate(mode, silero=lambda s: [(0, 1)]).regions(np.zeros(0, dtype=np.float32)) == []


def test_silero_is_what_decides_by_default():
    gate = ta.SpeechGate("silero", silero=lambda samples: [(10, 20)])
    assert gate.regions(np.zeros(100, dtype=np.float32)) == [(10, 20)]
    assert gate.describe() == "silero"


def test_a_silero_that_fails_falls_back_to_the_energy_gate_once_and_says_so(caplog):
    calls = []

    def broken(samples):
        calls.append(1)
        raise ImportError("no silero")

    gate = ta.SpeechGate("silero", silero=broken)
    audio = np.zeros(seconds(2), dtype=np.float32)
    audio[seconds(0.5):seconds(1.5)] = tone(1.0)
    assert gate.regions(audio), "the energy gate still finds the tone"
    gate.regions(audio)
    assert calls == [1], "a broken Silero is not retried on every request"
    assert gate.describe() == "energy (silero unavailable)"
    assert "energy gate" in caplog.text


def test_an_unknown_mode_is_silero():
    assert ta.SpeechGate("webrtc", silero=lambda s: []).mode == "silero"


# --- chunks ------------------------------------------------------------------------------------


def plan(regions, total_s=60.0, max_chunk_s=30.0, **kwargs):
    samples = np.zeros(seconds(total_s), dtype=np.float32)
    chunks = ta.plan_speech_chunks(samples, [(seconds(a), seconds(b)) for a, b in regions],
                                   max_chunk_s=max_chunk_s, **kwargs)
    return [(round(c.start / RATE, 2), round(c.end / RATE, 2)) for c in chunks]


def test_no_speech_is_no_chunk():
    assert plan([]) == []


def test_close_regions_share_a_chunk_and_the_pause_between_them_stays_in():
    assert plan([(2, 5), (6, 9), (10.5, 12)]) == [(2, 12)]


def test_a_long_pause_starts_a_new_chunk_and_is_never_transcribed():
    assert plan([(2, 5), (8, 9)]) == [(2, 5), (8, 9)]


def test_the_limit_starts_a_new_chunk_between_two_regions():
    chunks = plan([(2, 12), (13, 23), (24, 34), (35, 45)], max_chunk_s=30)
    assert chunks == [(2, 23), (24, 45)]
    assert all(b - a <= 30 for a, b in chunks)


def test_one_region_longer_than_the_limit_is_cut_at_quiet_points_into_pieces_within_it():
    samples = np.zeros(seconds(100), dtype=np.float32)
    samples[seconds(5):seconds(95)] = tone(90)
    chunks = ta.plan_speech_chunks(samples, [(seconds(5), seconds(95))], max_chunk_s=30)
    assert chunks[0].start == seconds(5) and chunks[-1].end == seconds(95)
    for previous, current in zip(chunks, chunks[1:]):
        assert current.start == previous.end, "the pieces of one stretch tile it"
    assert all(c.end - c.start <= seconds(30) for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_overlapping_and_unsorted_regions_are_merged_and_clipped_to_the_audio():
    assert plan([(10, 12), (3, 6), (5, 8), (55, 70)], total_s=60) == [(3, 12), (55, 60)]


def test_speech_near_the_start_or_the_end_of_the_file_is_extended_to_it():
    assert plan([(0.7, 3), (57.5, 59.2)], total_s=60) == [(0, 3), (57.5, 60)]
    assert plan([(1.5, 3)], total_s=60) == [(1.5, 3)], "1.5 s from the start is a real pause"
    assert plan([(0.7, 3)], total_s=60, edge_snap_s=0) == [(0.7, 3)]


def test_speech_seconds_counts_what_reaches_the_model():
    chunks = ta.plan_speech_chunks(np.zeros(seconds(60), dtype=np.float32),
                                   [(seconds(2), seconds(5)), (seconds(20), seconds(21))], max_chunk_s=30)
    assert ta.speech_seconds(chunks) == pytest.approx(4.0)


# --- segments ----------------------------------------------------------------------------------


def _chunks(*spans):
    return [ta.Chunk(i, seconds(a), seconds(b)) for i, (a, b) in enumerate(spans)]


def test_each_chunk_with_text_is_one_segment_with_the_chunks_times():
    segments, text, capped = ta.build_segments(
        _chunks((0.5, 4), (6, 9.25)), [ta.ChunkText("Guten Tag."), ta.ChunkText(" Wie geht es? ")])
    assert segments == [{"start": 0.5, "end": 4.0, "text": "Guten Tag."},
                        {"start": 6.0, "end": 9.25, "text": "Wie geht es?"}]
    assert text == "Guten Tag. Wie geht es?"
    assert capped == []


def test_a_chunk_without_text_leaves_no_segment():
    segments, text, _ = ta.build_segments(_chunks((0, 2), (3, 4)), [ta.ChunkText(""), ta.ChunkText("ja")])
    assert [s["text"] for s in segments] == ["ja"] and text == "ja"


def test_a_capped_chunk_is_flagged_and_named():
    segments, _, capped = ta.build_segments(_chunks((0, 30)), [ta.ChunkText("lang", capped=True)])
    assert segments[0]["truncated"] is True
    assert capped == ["0-30 s"]


def test_chinese_and_japanese_pieces_are_joined_without_spaces():
    _, text, _ = ta.build_segments(_chunks((0, 2), (3, 4)), [ta.ChunkText("你好"), ta.ChunkText("世界")])
    assert text == "你好世界"


def test_a_transcript_count_that_does_not_match_the_chunks_is_an_error():
    with pytest.raises(ValueError):
        ta.build_segments(_chunks((0, 2)), [])


# --- Cohere specifics ----------------------------------------------------------------------------


def test_the_mirror_is_pinned_and_held_to_the_official_digest_and_size():
    assert cs.default_revision(cs.MIRROR_MODEL) == cs.MIRROR_REVISION
    assert len(cs.MIRROR_REVISION) == 40
    assert cs.default_sha256(cs.MIRROR_MODEL) == cs.OFFICIAL_WEIGHTS_SHA256
    assert cs.expected_size(cs.OFFICIAL_WEIGHTS_SHA256) == 4131862976


def test_the_official_repository_and_anything_else_follow_main_without_a_default_digest():
    for model in (cs.OFFICIAL_MODEL, "someone/fine-tune", "/models/cohere"):
        assert cs.default_revision(model) == "main"
        assert cs.default_sha256(model) is None
    assert cs.expected_size("0" * 64) is None


def test_the_fourteen_languages_of_the_model_card():
    assert cs.SUPPORTED_LANGUAGES == {"ar", "de", "el", "en", "es", "fr", "it", "ja", "ko", "nl", "pl", "pt", "vi", "zh"}
    assert cs.MAX_CHUNK_S == 30.0


def test_generated_lengths_count_new_tokens_up_to_the_end_or_padding():
    prompt = [9] * 10
    rows = [prompt + [11, 12, 3, 2, 2], prompt + [11, 12, 13, 14, 15]]
    assert cs.generated_lengths(rows, 10, pad_token_id=2, eos_token_id=3) == [2, 5]


# --- Granite specifics -----------------------------------------------------------------------------


def test_both_ibm_models_are_pinned_to_the_measured_commits():
    assert gs.default_revision(gs.DEFAULT_MODEL) == "f4e68169fa2a5751acc4b5da9957855e5e8fc3f1"
    assert gs.default_revision(gs.TURBOCTC_MODEL) == "c85a3307b6abec4a169daaf11e8dfa80ac27ad81"
    assert gs.default_revision("someone/granite-ft") == "main"


def test_the_kind_of_checkpoint_comes_from_its_model_type():
    assert gs.family_for("granite_speech") == gs.FAMILY_LLM
    assert gs.family_for("granite_speech5_ctc") == gs.FAMILY_CTC
    with pytest.raises(ValueError, match="cohere_asr"):
        gs.family_for("cohere_asr")
    with pytest.raises(ValueError):
        gs.family_for(None)


def test_each_model_accepts_its_own_languages_and_the_override_wins():
    assert gs.supported_languages(gs.DEFAULT_MODEL, gs.FAMILY_LLM) == {"en", "fr", "de", "es", "pt", "ja"}
    assert gs.supported_languages(gs.TURBOCTC_MODEL, gs.FAMILY_CTC) == {"en", "fr", "de", "es", "pt", "it", "ja"}
    assert gs.supported_languages("/models/ft", gs.FAMILY_CTC) == gs.LANGUAGES_BY_FAMILY[gs.FAMILY_CTC]
    assert gs.supported_languages(gs.DEFAULT_MODEL, gs.FAMILY_LLM, "de, en-GB;nl") == {"de", "en", "nl"}
    assert gs.supported_languages(gs.DEFAULT_MODEL, gs.FAMILY_LLM, " , ") == gs.KNOWN_MODELS[gs.DEFAULT_MODEL]["languages"]


@pytest.mark.parametrize("values, expected", [
    (("ACME, Leipzig, Grafana",), ["ACME", "Leipzig", "Grafana"]),
    (("ACME;Leipzig\nGrafana\r\nHelm",), ["ACME", "Leipzig", "Grafana", "Helm"]),
    (("  St.  Louis ,, ",), ["St. Louis"]),
    (("ACME, acme, Acme",), ["ACME"]),
    (("ACME", "Leipzig, acme"), ["ACME", "Leipzig"]),
    ((None, ""), []),
])
def test_keyword_lists_are_split_trimmed_and_deduplicated(values, expected):
    assert gs.parse_keywords(*values) == expected


def test_more_keywords_than_the_model_takes_are_refused_not_cut():
    assert len(gs.parse_keywords(",".join(f"w{i}" for i in range(2000)))) == 2000
    with pytest.raises(gs.KeywordError, match="2000"):
        gs.parse_keywords(",".join(f"w{i}" for i in range(2001)))


def test_an_overlong_keyword_is_refused_with_a_hint():
    with pytest.raises(gs.KeywordError, match="commas"):
        gs.parse_keywords("x" * 101)


def test_the_prompt_names_the_language_asks_for_punctuation_and_lists_keywords():
    assert gs.build_prompt("de") == (
        "<|audio|>The speech is in German. Transcribe it exactly; do not translate. "
        "transcribe the speech with proper punctuation and capitalization.")
    assert gs.build_prompt(None) == "<|audio|>transcribe the speech with proper punctuation and capitalization."
    assert gs.build_prompt("pt", ["ACME", "Leipzig"]).endswith(
        "capitalization. Keywords: ACME, Leipzig")
    assert "Italian" in gs.build_prompt("it")
