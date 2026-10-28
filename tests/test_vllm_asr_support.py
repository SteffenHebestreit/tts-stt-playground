"""vllm-asr-service's pure helpers: the model catalogue, the engine's command line, one chunk as a request.

vllm_support.py imports no torch and no vLLM, so it is tested directly. The engine's
lifecycle is in test_vllm_asr_engine.py, the app end to end in test_vllm_asr_service.py.
"""

from __future__ import annotations

import base64
import io
import wave

import numpy as np
import pytest

from vllm_asr_loader import load_module

support = load_module("vllm_support")
cohere = load_module("cohere_support")
granite = load_module("granite_support")


# --- which model, which family ------------------------------------------------------------------


@pytest.mark.parametrize("model_type, family", [
    ("whisper", "whisper"), ("cohere_asr", "cohere"), ("qwen3_asr", "qwen3-asr"), ("granite_speech", "granite"),
])
def test_the_family_comes_from_the_model_type(model_type, family):
    assert support.family_for(model_type) == family


@pytest.mark.parametrize("model_type", ["granite_speech5_ctc", "llama", "", None])
def test_a_checkpoint_of_another_kind_is_refused_with_the_supported_ones_named(model_type):
    with pytest.raises(support.ModelChoiceError, match="cohere_asr"):
        support.family_for(model_type)


def test_the_offered_models_are_the_measured_ones_each_pinned_to_its_commit():
    assert support.DEFAULT_MODEL == "ibm-granite/granite-speech-5.0-2b"
    assert set(support.OFFERED_MODELS) <= set(support.KNOWN_MODELS)
    for model in support.OFFERED_MODELS:
        revision = support.default_revision(model)
        assert len(revision) == 40 and int(revision, 16) >= 0, model
    assert support.default_revision("someone/fine-tune") == "main"


def test_granite_and_cohere_are_pinned_exactly_as_their_own_services_pin_them():
    """Both services then share one download in a common cache (the TrueNAS app's)."""
    assert support.default_revision(granite.DEFAULT_MODEL) == granite.KNOWN_MODELS[granite.DEFAULT_MODEL]["revision"]
    assert support.default_revision(cohere.MIRROR_MODEL) == cohere.MIRROR_REVISION
    assert support.default_sha256(cohere.MIRROR_MODEL) == cohere.OFFICIAL_WEIGHTS_SHA256
    assert support.KNOWN_MODELS[cohere.MIRROR_MODEL].size == cohere.OFFICIAL_WEIGHTS_SIZE
    # The official repository is the source of truth itself, and nothing else has a digest.
    assert support.default_sha256(cohere.OFFICIAL_MODEL) is None
    assert support.default_sha256(support.QWEN3_ASR) is None


def test_the_languages_are_the_models_own_and_never_more_than_vllm_takes():
    assert support.supported_languages(granite.DEFAULT_MODEL, "granite") == {"de", "en", "fr", "es", "pt", "ja"}
    assert support.supported_languages(cohere.MIRROR_MODEL, "cohere") == cohere.SUPPORTED_LANGUAGES
    assert support.supported_languages(support.WHISPER_GERMAN, "whisper") == {"de"}
    qwen = support.supported_languages(support.QWEN3_ASR, "qwen3-asr")
    assert "de" in qwen and len(qwen) == 28 and qwen <= support.VLLM_LANGUAGES
    # Cantonese is Qwen3-ASR's but not in vLLM 0.31's list: sending it would be a 400.
    assert "yue" not in qwen
    # A Whisper fine-tune of unknown provenance: Whisper's languages, as vLLM lists them.
    assert support.supported_languages("someone/whisper-ft", "whisper") == support.VLLM_LANGUAGES


def test_the_language_override_replaces_the_list_but_cannot_exceed_vllms():
    assert support.supported_languages(support.WHISPER_GERMAN, "whisper", "de, en_US;fr") == {"de", "en", "fr"}
    assert support.supported_languages(support.QWEN3_ASR, "qwen3-asr", "de yue") == {"de"}
    assert support.supported_languages(granite.DEFAULT_MODEL, "granite", "  ") == {"de", "en", "fr", "es", "pt", "ja"}


def test_only_whisper_multilingual_and_qwen3_identify_the_language():
    assert support.detects_language(support.QWEN3_ASR, "qwen3-asr") is True
    assert support.detects_language("openai/whisper-large-v3-turbo", "whisper") is True
    # The German fine-tune is told German; it is not asked which language it hears.
    assert support.detects_language(support.WHISPER_GERMAN, "whisper") is False
    assert support.detects_language(cohere.MIRROR_MODEL, "cohere") is False
    assert support.detects_language(granite.DEFAULT_MODEL, "granite") is False


def test_granite_goes_through_the_chat_endpoint_and_the_others_through_transcriptions():
    assert support.endpoint_for("granite") == support.ENDPOINT_CHAT
    for family in ("whisper", "cohere", "qwen3-asr"):
        assert support.endpoint_for(family) == support.ENDPOINT_TRANSCRIPTIONS


# --- the engine's command line -----------------------------------------------------------------------


def test_the_engine_listens_on_loopback_with_the_configured_share_of_the_gpu():
    argv = support.engine_argv("/models/x", "ibm-granite/granite-speech-5.0-2b", 8100,
                               support.EngineSettings(gpu_memory_utilization=0.45), default_max_model_len=2048)
    assert argv[:3] == ["vllm", "serve", "/models/x"]
    joined = " ".join(argv)
    assert "--host 127.0.0.1" in joined and "--port 8100" in joined
    assert "--served-model-name ibm-granite/granite-speech-5.0-2b" in joined
    assert "--gpu-memory-utilization 0.45" in joined and "--max-model-len 2048" in joined
    assert "--enable-sleep-mode" in argv
    assert "--trust-remote-code" not in argv


def test_the_settings_win_over_the_models_default_and_extra_arguments_come_last():
    settings = support.EngineSettings(gpu_memory_utilization=0.5, max_model_len=4096, max_num_seqs=64,
                                      max_num_batched_tokens=2048, sleep_mode=False,
                                      extra_args=support.parse_extra_args("--enforce-eager --dtype 'bfloat16'"))
    argv = support.engine_argv("/m", "m", 8100, settings, default_max_model_len=2048)
    joined = " ".join(argv)
    assert "--max-model-len 4096" in joined and "--max-model-len 2048" not in joined
    assert "--max-num-seqs 64" in joined and "--max-num-batched-tokens 2048" in joined
    assert "--enable-sleep-mode" not in argv
    assert argv[-3:] == ["--enforce-eager", "--dtype", "bfloat16"]


def test_without_a_default_the_engine_uses_the_models_own_context():
    argv = support.engine_argv("/m", "m", 8100, support.EngineSettings())
    assert "--max-model-len" not in argv


@pytest.mark.parametrize("extra, colliding", [
    ("--port 9000", ["--port"]), ("--gpu-memory-utilization=0.9", ["--gpu-memory-utilization=0.9"]),
    ("--trust-remote-code", ["--trust-remote-code"]), ("--enforce-eager --swap-space 4", []),
])
def test_extra_arguments_may_not_repeat_the_services_own(extra, colliding):
    assert support.check_extra_args(support.parse_extra_args(extra)) == colliding


# --- one chunk as a request ---------------------------------------------------------------------------


def test_a_chunk_reaches_the_engine_as_16_bit_wav_bit_for_bit():
    original = (np.arange(-2000, 2000, dtype=np.int16) * 8)
    samples = original.astype(np.float32) / 32768.0
    data = support.wav_bytes(samples)
    with wave.open(io.BytesIO(data), "rb") as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
        back = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    assert np.array_equal(back, original)


def test_a_chunk_beyond_full_scale_is_clipped_not_wrapped():
    data = support.wav_bytes(np.array([1.5, -1.5, 1.0, -1.0], dtype=np.float32))
    with wave.open(io.BytesIO(data), "rb") as wav:
        back = np.frombuffer(wav.readframes(4), dtype="<i2")
    assert back.tolist() == [32767, -32768, 32767, -32768]


def test_the_prompt_is_whispers_initial_prompt_and_qwens_context_and_nothing_for_the_others():
    assert support.prompt_text("whisper", " Leipzig, ACME ", []) == "Leipzig, ACME"
    assert support.prompt_text("qwen3-asr", None, ["ACME", "Leipzig"]) == "ACME, Leipzig"
    assert support.prompt_text("qwen3-asr", "Ein Gespräch", ["ACME"]) == "Ein Gespräch"
    assert support.prompt_text("cohere", "anything", ["x"]) == ""
    assert support.prompt_text("granite", "anything", ["x"]) == ""


def test_a_transcription_request_is_greedy_capped_and_names_the_language_only_when_known():
    fields = support.transcription_fields("m", "de", "ACME", 560)
    assert fields == {"model": "m", "response_format": "json", "temperature": "0",
                      "max_completion_tokens": "560", "language": "de", "prompt": "ACME"}
    assert "language" not in support.transcription_fields("m", None, "", 560)
    assert "prompt" not in support.transcription_fields("m", "de", "", 560)


def test_a_granite_chunk_carries_the_audio_first_and_granite_asr_services_prompt():
    body = support.chat_body("ibm-granite/granite-speech-5.0-2b", b"RIFF....", "de", ["ACME", "Leipzig"], 560)
    audio, text = body["messages"][0]["content"]
    assert audio["type"] == "input_audio" and audio["input_audio"]["format"] == "wav"
    assert base64.b64decode(audio["input_audio"]["data"]) == b"RIFF...."
    expected = granite.build_prompt("de", ["ACME", "Leipzig"])
    assert expected.startswith("<|audio|>") and text["text"] == expected[len("<|audio|>"):]
    assert "German" in text["text"] and "Keywords: ACME, Leipzig" in text["text"]
    assert body["temperature"] == 0 and body["max_tokens"] == 560


def test_whispers_leading_space_is_dropped_and_an_answer_without_text_is_an_error():
    assert support.parse_transcription({"text": " Als Bettler ward ich", "usage": {}}).text == "Als Bettler ward ich"
    with pytest.raises(ValueError):
        support.parse_transcription({"error": {"message": "no"}})


def test_a_chat_answer_that_stopped_on_the_token_cap_is_flagged():
    stopped = support.parse_chat({"choices": [{"message": {"content": " Hallo. "}, "finish_reason": "stop"}]})
    capped = support.parse_chat({"choices": [{"message": {"content": "Hallo"}, "finish_reason": "length"}]})
    assert (stopped.text, stopped.capped) == ("Hallo.", False)
    assert capped.capped is True
    assert support.parse_chat({"choices": [{"message": {"content": None}, "finish_reason": "stop"}]}).text == ""
    with pytest.raises(ValueError):
        support.parse_chat({"choices": []})
