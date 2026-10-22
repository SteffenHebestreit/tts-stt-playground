"""cohere-asr-service and granite-asr-service end to end, under stand-ins for the models.

The apps run for real (FastAPI, upload spooling, decoding through the fake ffmpeg,
the speech gate, chunk planning, batching, the lifecycle slot); only torch,
transformers, huggingface_hub and silero_vad are fakes (tests/transformers_asr_loader.py).
The fake models "hear" words: a run of constant amplitude is ``w<level>``, so a
lossless pipeline must return exactly ``expected_transcript``.
"""

from __future__ import annotations

import asyncio
import hashlib

import httpx
import numpy as np
import pytest

from test_qwen3_asr_harness import (
    FakeFfmpeg, encoded_wav, expected_transcript, silent_wav, speech_levels, wav_from_int16,
)
from transformers_asr_loader import WEIGHTS, WAIT, load_app

FAKE_SHA = hashlib.sha256(WEIGHTS).hexdigest()
OK_WEIGHTS = {"COHERE_ASR_SHA256": FAKE_SHA}


def client(module) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=module.app, raise_app_exceptions=False)
    return httpx.AsyncClient(transport=transport, base_url="http://asr.test", timeout=30.0)


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def service(monkeypatch, tmp_path):
    """``service(name, **env)``: a loaded app with the fake ffmpeg installed."""
    ffmpeg = FakeFfmpeg(tmp_path)

    def build(name, **env):
        module = load_app(name, monkeypatch, **env)
        ffmpeg.install(module, monkeypatch)
        return module

    return build


def post(module, wav: bytes, **fields):
    async def call():
        async with client(module) as c:
            return await c.post("/transcribe", data=fields, files={"audio": ("a.wav", wav, "audio/wav")})
    return run(call())


def get(module, path):
    async def call():
        async with client(module) as c:
            return await c.get(path)
    return run(call())


def words_wav(words: int, pause_every: int = 8) -> tuple[bytes, list[int]]:
    levels = speech_levels(words, pause_every=pause_every)
    return encoded_wav(levels), levels


# --- Cohere ------------------------------------------------------------------------------


def test_cohere_transcribes_in_the_requested_language_with_one_segment_per_chunk(service):
    module = service("cohere", **OK_WEIGHTS)
    wav, levels = words_wav(20)
    r = post(module, wav, language="de")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"] == expected_transcript(levels)
    assert body["language"] == "de" and body["warnings"] == [] and body["truncated"] is False
    assert body["segments"] and all(s["end"] > s["start"] for s in body["segments"])
    assert body["duration"] == pytest.approx(len(levels) * 0.25, abs=0.01)
    assert body["model"] == "evewashere/cohere-transcribe-03-2026-ungated"
    calls = module._fake.processor_calls
    assert {c["language"] for c in calls} == {"de"} and {c["sampling_rate"] for c in calls} == {16000}
    assert all(c["punctuation"] is True for c in calls)


def test_cohere_fetches_the_pinned_mirror_with_the_allow_list_and_no_token(service):
    module = service("cohere", **OK_WEIGHTS)
    post(module, words_wav(4)[0], language="de")
    (download,) = module._fake.downloads
    assert download["repo_id"] == "evewashere/cohere-transcribe-03-2026-ungated"
    assert download["revision"] == "29b9036c65620e1a148127c6147543b52358da6a"
    assert "model.safetensors" in download["allow_patterns"] and download["token"] is None
    assert not any(p.endswith(".py") for p in download["allow_patterns"])


def test_the_official_repository_follows_main_and_gets_the_token(service):
    module = service("cohere", COHERE_ASR_MODEL="CohereLabs/cohere-transcribe-03-2026", HF_TOKEN="hf_secret")
    r = post(module, words_wav(4)[0], language="de")
    assert r.status_code == 200, r.text
    (download,) = module._fake.downloads
    assert download["revision"] == "main" and download["token"] == "hf_secret"
    status = get(module, "/status").json()
    assert status["weights_check"] == "off"
    assert "hf_secret" not in get(module, "/status").text + get(module, "/health").text


def test_the_mirrors_weights_must_match_the_official_digest(service):
    module = service("cohere")   # the fake weights are not Cohere's
    r = post(module, words_wav(4)[0], language="de")
    assert r.status_code == 503 and "could not be loaded" in r.json()["detail"]
    assert "safetensors" not in r.text and "sha" not in r.text.lower()
    ready = get(module, "/ready")
    assert ready.status_code == 503 and ready.json()["reason"] == "load_failed"
    status = get(module, "/status").json()
    assert status["weights_check"] == "failed" and status["model_loaded"] is False
    assert module._fake.generate_calls == []
    assert not [load for load in module._fake.loads if load["kind"] == "model"], "nothing may read unchecked weights"


def test_a_digest_setting_that_is_no_digest_stops_the_load_instead_of_skipping_the_check(service):
    module = service("cohere", COHERE_ASR_SHA256="not-a-digest")
    assert post(module, words_wav(4)[0], language="de").status_code == 503
    assert get(module, "/ready").json()["reason"] == "load_failed"
    assert module._fake.downloads == []


def test_the_check_can_be_switched_off(service):
    module = service("cohere", COHERE_ASR_SHA256="off")
    assert post(module, words_wav(4)[0], language="de").status_code == 200
    assert get(module, "/status").json()["weights_check"] == "off"


def test_a_mounted_checkpoint_is_not_downloaded_and_its_path_is_not_shown(service, tmp_path):
    module = service("cohere")
    local = module._fake.model_dir
    module = service("cohere", COHERE_ASR_MODEL=str(local))
    module._fake.model_dir = local
    assert post(module, words_wav(4)[0], language="de").status_code == 200
    assert module._fake.downloads == []
    assert str(local) not in get(module, "/status").text


@pytest.mark.parametrize("language", ["sv", "German", "xx"])
def test_cohere_refuses_a_language_it_does_not_know_before_loading_anything(service, language):
    module = service("cohere", **OK_WEIGHTS)
    r = post(module, words_wav(4)[0], language=language)
    assert r.status_code == 422
    assert "Supported: ar, de, el, en, es, fr, it, ja, ko, nl, pl, pt, vi, zh" in r.json()["detail"]
    assert module._fake.downloads == [] and module._fake.generate_calls == []


def test_auto_means_the_default_language_and_says_so_while_no_language_says_nothing(service):
    module = service("cohere", COHERE_ASR_DEFAULT_LANGUAGE="fr", **OK_WEIGHTS)
    auto = post(module, words_wav(4)[0], language="auto").json()
    missing = post(module, words_wav(4)[0]).json()
    assert auto["language"] == missing["language"] == "fr"
    assert "cannot detect the language" in auto["warnings"][0]
    assert missing["warnings"] == []


def test_a_prompt_is_ignored_with_a_warning(service):
    module = service("cohere", **OK_WEIGHTS)
    body = post(module, words_wav(4)[0], language="de", initial_prompt="LUNA, Bautzen").json()
    assert any("takes no prompt" in w for w in body["warnings"])


def test_silence_never_reaches_the_model(service):
    module = service("cohere", **OK_WEIGHTS)
    body = post(module, silent_wav(8), language="de").json()
    assert body["text"] == "" and body["segments"] == [] and body["chunks"] == 0
    assert body["speech_seconds"] == 0
    assert "No speech" in body["warnings"][0]
    assert module._fake.generate_calls == []


def test_a_long_recording_is_cut_in_its_pauses_into_batched_chunks_without_losing_a_word(service):
    module = service("cohere", COHERE_ASR_CHUNK_S="10", COHERE_ASR_BATCH_SIZE="2", **OK_WEIGHTS)
    # 160 words of 0.25 s with a 0.25 s pause after every 8: 45 s of speech.
    wav, levels = words_wav(160, pause_every=8)
    body = post(module, wav, language="de").json()
    assert body["text"] == expected_transcript(levels)
    assert body["chunks"] >= 5
    assert all(s["end"] - s["start"] <= 10.0 + 1e-6 for s in body["segments"])
    starts = [s["start"] for s in body["segments"]]
    assert starts == sorted(starts)
    assert all(call["pieces"] <= 2 for call in module._fake.generate_calls)
    assert sum(call["pieces"] for call in module._fake.generate_calls) == body["chunks"]


def test_a_chunk_that_hits_the_token_cap_is_flagged(service):
    module = service("cohere", COHERE_ASR_MAX_NEW_TOKENS="32", **OK_WEIGHTS)
    wav, levels = words_wav(60, pause_every=0)   # one stretch, 15 s, 60 words
    body = post(module, wav, language="de").json()
    assert body["truncated"] is True and body["segments"][0]["truncated"] is True
    assert any("32-token limit" in w for w in body["warnings"])
    assert len(body["text"].split()) == 32


def test_a_broken_silero_falls_back_to_the_energy_gate(service):
    module = service("cohere", **OK_WEIGHTS)
    module._fake.vad_error = RuntimeError("jit model is corrupt")
    wav, levels = words_wav(12)
    body = post(module, wav, language="de").json()
    assert body["text"] == expected_transcript(levels)
    assert get(module, "/status").json()["vad"] == "energy (silero unavailable)"


def test_health_and_status_answer_without_loading_the_model(service):
    module = service("cohere", **OK_WEIGHTS)
    health = get(module, "/health").json()
    assert health["model_loaded"] is False and health["model_resident"] is False
    assert health["default_language"] == "de" and health["device"] == "cpu"
    status = get(module, "/status").json()
    assert status["supported_languages"] == ["ar", "de", "el", "en", "es", "fr", "it", "ja", "ko", "nl",
                                             "pl", "pt", "vi", "zh"]
    assert status["language_detection"] is False and status["timestamps"] == "chunk"
    assert status["weights_sha256"] == FAKE_SHA and status["weights_check"] == "pending"
    assert status["chunk_seconds"] == 30.0 and status["max_new_tokens"] == 600 and status["vad"] == "silero"
    assert module._fake.downloads == [] and module._fake.loads == []


def test_the_chunk_length_cannot_exceed_what_the_model_takes_whole(service):
    module = service("cohere", COHERE_ASR_CHUNK_S="120", **OK_WEIGHTS)
    assert module.CHUNK_S == 30.0


def test_unload_refuses_while_a_transcription_runs_and_frees_the_model_after(service):
    import threading

    module = service("cohere", **OK_WEIGHTS)
    module._fake.generate_gate = threading.Event()

    async def scenario():
        async with client(module) as c:
            busy = asyncio.create_task(
                c.post("/transcribe", data={"language": "de"}, files={"audio": ("a.wav", words_wav(6)[0])}))
            while not module._fake.generate_entered.is_set():
                await asyncio.sleep(0.01)
            refused = await c.post("/unload")
            module._fake.generate_gate.set()
            done = await asyncio.wait_for(busy, WAIT)
            freed = await c.post("/unload")
            health = await c.get("/health")
            return refused, done, freed, health

    refused, done, freed, health = run(scenario())
    assert refused.status_code == 409
    assert done.status_code == 200
    assert freed.status_code == 200 and freed.json()["model_resident"] is False
    assert health.json()["model_resident"] is False and health.json()["model_loaded"] is False


def test_an_empty_upload_and_an_unreadable_one_are_client_errors(service):
    module = service("cohere", **OK_WEIGHTS)
    assert post(module, b"", language="de").status_code == 400
    assert post(module, b"this is not audio at all", language="de").status_code == 422


# --- Granite ------------------------------------------------------------------------------


def test_granite_names_the_language_and_the_keywords_in_its_prompt(service):
    module = service("granite")
    wav, levels = words_wav(16)
    body = post(module, wav, language="de", keywords="LUNA, Bautzen").json()
    assert body["text"] == expected_transcript(levels)
    assert body["language"] == "de" and body["keywords"] == 2 and body["warnings"] == []
    prompts = {p for call in module._fake.processor_calls for p in call["prompts"]}
    assert prompts == {"USER: <|audio|>The speech is in German. Transcribe it exactly; do not translate. "
                       "transcribe the speech with proper punctuation and capitalization. Keywords: LUNA, Bautzen"
                       "\n ASSISTANT:"}
    (download,) = module._fake.downloads
    assert download["repo_id"] == "ibm-granite/granite-speech-5.0-2b"
    assert download["revision"] == "f4e68169fa2a5751acc4b5da9957855e5e8fc3f1"


def test_the_openai_prompt_is_read_as_a_keyword_list_too(service):
    module = service("granite")
    body = post(module, words_wav(6)[0], language="de", initial_prompt="Spot; Sphere\nStation",
                keywords="Spot").json()
    assert body["keywords"] == 3
    prompt = module._fake.processor_calls[0]["prompts"][0]
    assert prompt.rstrip().endswith("Keywords: Spot, Sphere, Station\n ASSISTANT:".rstrip())


def test_too_many_keywords_are_refused_before_anything_loads(service):
    module = service("granite")
    r = post(module, words_wav(4)[0], language="de", keywords=",".join(f"k{i}" for i in range(2001)))
    assert r.status_code == 422 and "2000" in r.json()["detail"]
    assert module._fake.downloads == []


def test_the_2b_refuses_italian_and_the_turboctc_takes_it(service):
    llm = service("granite")
    assert post(llm, words_wav(4)[0], language="it").status_code == 422
    ctc = service("granite", GRANITE_ASR_MODEL="ibm-granite/granite-speech-5.0-470m-turboctc")
    wav, levels = words_wav(10)
    body = post(ctc, wav, language="it").json()
    assert body["language"] == "it" and body["text"] == expected_transcript(levels)


def test_the_turboctc_takes_no_keywords_and_says_so(service):
    module = service("granite", GRANITE_ASR_MODEL="ibm-granite/granite-speech-5.0-470m-turboctc")
    body = post(module, words_wav(6)[0], language="de", keywords="LUNA").json()
    assert body["keywords"] == 0
    assert any("takes no keywords" in w for w in body["warnings"])
    assert all(call["max_new_tokens"] is None for call in module._fake.generate_calls)
    status = get(module, "/status").json()
    assert status["model_family"] == "ctc" and status["keywords"] is False and status["max_new_tokens"] is None
    assert "it" in status["supported_languages"]
    (download,) = module._fake.downloads
    assert download["revision"] == "c85a3307b6abec4a169daaf11e8dfa80ac27ad81"


def test_an_unknown_checkpoint_takes_its_kind_and_languages_from_its_config(service):
    module = service("granite", GRANITE_ASR_MODEL="someone/granite-ctc-ft", model_type="granite_speech5_ctc")
    assert get(module, "/status").json()["model_family"] is None
    assert post(module, words_wav(4)[0], language="de").status_code == 200
    status = get(module, "/status").json()
    assert status["model_family"] == "ctc" and "it" in status["supported_languages"]
    assert module._fake.downloads[0]["revision"] == "main"


def test_a_checkpoint_of_another_kind_is_refused_at_load(service):
    module = service("granite", GRANITE_ASR_MODEL="someone/other", model_type="whisper")
    assert post(module, words_wav(4)[0], language="de").status_code == 503
    assert get(module, "/ready").json()["reason"] == "load_failed"


def test_granite_can_require_a_digest(service):
    bad = service("granite", GRANITE_ASR_SHA256="0" * 64)
    assert post(bad, words_wav(4)[0], language="de").status_code == 503
    assert get(bad, "/status").json()["weights_check"] == "failed"
    good = service("granite", GRANITE_ASR_SHA256=FAKE_SHA)
    assert post(good, words_wav(4)[0], language="de").status_code == 200
    assert get(good, "/status").json()["weights_check"] == "ok"


def test_granite_status_reports_what_the_gateway_reads(service):
    module = service("granite", GRANITE_ASR_DEFAULT_LANGUAGE="fr")
    status = get(module, "/status").json()
    assert status["supported_languages"] == ["de", "en", "es", "fr", "ja", "pt"]
    assert status["default_language"] == "fr" and status["current_model"] == "ibm-granite/granite-speech-5.0-2b"
    assert get(module, "/health").json()["default_language"] == "fr"


def test_granite_auto_is_the_default_language_with_a_warning(service):
    module = service("granite")
    body = post(module, words_wav(4)[0], language="auto").json()
    assert body["language"] == "de" and "not asked to detect" in body["warnings"][0]


def test_granite_gates_silence_too(service):
    module = service("granite")
    body = post(module, wav_from_int16(np.zeros(16000 * 6, dtype=np.int16)), language="de").json()
    assert body["text"] == "" and module._fake.generate_calls == []


def test_a_failure_after_a_good_load_is_still_a_500(service, monkeypatch):
    """Only a load that failed is a 503; a transcription that breaks is the service's fault."""
    module = service("cohere", **OK_WEIGHTS)
    assert post(module, words_wav(4)[0], language="de").status_code == 200

    def broken(*args, **kwargs):
        raise RuntimeError("index out of range in the decoder")

    monkeypatch.setattr(module, "_run_model", broken)
    r = post(module, words_wav(4)[0], language="de")
    assert r.status_code == 500 and "Request id" in r.json()["detail"] and "decoder" not in r.text


@pytest.mark.parametrize("name, env", [("cohere", OK_WEIGHTS), ("granite", {})])
def test_a_load_that_runs_out_of_gpu_memory_gives_the_memory_back(service, name, env):
    """An OOM half-way through .to("cuda") must not leave the moved part reserved.

    On the NAS (2026-10-22) both services kept 2.3-3.3 GB after such a load while
    reporting "not resident": the traceback still held the model, so the gateway's
    own empty_cache() ran while it was alive. The loader drops it before that.
    """
    module = service(name, **env)
    module.device = "cuda"
    module._fake.to_error = RuntimeError("CUDA error: out of memory")
    r = post(module, words_wav(4)[0], language="de")
    assert r.status_code == 503
    assert any(module._fake.empty_cache_calls), "empty_cache never ran after the model was gone"
    assert get(module, "/status").json().get("model_loaded") is not True

    module._fake.to_error = None
    assert post(module, words_wav(4)[0], language="de").status_code == 200
