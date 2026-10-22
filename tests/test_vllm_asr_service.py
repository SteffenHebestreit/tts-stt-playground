"""vllm-asr-service end to end, under a stand-in for ``vllm serve``.

The app runs for real (FastAPI, upload spooling, decoding, the speech gate, chunk
planning, the lifecycle slot, the engine's process handling and its HTTP calls); only
the engine process and its HTTP API, huggingface_hub and silero_vad are fakes
(tests/vllm_asr_loader.py). The fake engine "hears" words, so a lossless pipeline
returns exactly ``expected_transcript``.
"""

from __future__ import annotations

import asyncio
import hashlib
import threading

import httpx
import pytest

from test_qwen3_asr_harness import encoded_wav, expected_transcript, silent_wav, speech_levels
from vllm_asr_loader import WAIT, WEIGHTS, load_app

FAKE_SHA = hashlib.sha256(WEIGHTS).hexdigest()
COHERE = {"VLLM_ASR_MODEL": "evewashere/cohere-transcribe-03-2026-ungated", "VLLM_ASR_SHA256": FAKE_SHA}
QWEN = {"VLLM_ASR_MODEL": "Qwen/Qwen3-ASR-1.7B"}
WHISPER = {"VLLM_ASR_MODEL": "primeline/whisper-large-v3-turbo-german"}


@pytest.fixture
def service(monkeypatch):
    """``service(**env)``: a loaded app with the fake engine."""
    return lambda **env: load_app(monkeypatch, **env)


def client(module) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=module.app, raise_app_exceptions=False)
    return httpx.AsyncClient(transport=transport, base_url="http://asr.test", timeout=30.0)


def run(coro):
    return asyncio.run(coro)


def post(module, wav: bytes, **fields):
    async def call():
        async with client(module) as c:
            return await c.post("/transcribe", data=fields, files={"audio": ("a.wav", wav, "audio/wav")})
    return run(call())


def call(module, method: str, path: str, **kwargs):
    async def go():
        async with client(module) as c:
            return await c.request(method, path, **kwargs)
    return run(go())


def words_wav(words: int, pause_every: int = 8) -> tuple[bytes, list[int]]:
    levels = speech_levels(words, pause_every=pause_every)
    return encoded_wav(levels), levels


# --- the default model, Granite, through the chat endpoint --------------------------------------------


def test_granite_transcribes_through_the_chat_endpoint_with_its_own_prompt(service):
    module = service()
    wav, levels = words_wav(20)
    r = post(module, wav, language="de", initial_prompt="LUNA, Bautzen")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"] == expected_transcript(levels)
    assert body["language"] == "de" and body["model"] == "ibm-granite/granite-speech-5.0-2b"
    assert body["keywords"] == 2 and body["engine"] == "vllm" and body["warnings"] == []
    assert body["segments"] and all(s["end"] > s["start"] for s in body["segments"])
    requests = module._fake.requests
    assert requests and {r["path"] for r in requests} == {"/v1/chat/completions"}
    assert all(r["order"] == ["input_audio", "text"] for r in requests)
    assert all("The speech is in German" in r["prompt"] and "Keywords: LUNA, Bautzen" in r["prompt"]
               for r in requests)
    assert all(not r["prompt"].startswith("<|audio|>") for r in requests)


def test_the_checkpoint_is_fetched_pinned_with_the_allow_list_and_served_from_its_directory(service):
    module = service()
    post(module, words_wav(4)[0], language="de")
    (download,) = module._fake.downloads
    assert download["repo_id"] == "ibm-granite/granite-speech-5.0-2b"
    assert download["revision"] == "f4e68169fa2a5751acc4b5da9957855e5e8fc3f1"
    assert "model.safetensors" in download["allow_patterns"] and download["token"] is None
    assert not any(p.endswith(".py") for p in download["allow_patterns"])
    (process,) = module._fake.processes
    argv = process.argv
    assert argv[:3] == ["vllm", "serve", str(module._fake.model_dir)]
    assert "--max-model-len" in argv and argv[argv.index("--max-model-len") + 1] == "2048"
    assert argv[argv.index("--gpu-memory-utilization") + 1] == "0.45"
    assert "--enable-sleep-mode" in argv and argv[argv.index("--host") + 1] == "127.0.0.1"


def test_the_engine_gets_the_gpu_back_but_no_token_and_no_network(service):
    module = service(CUDA_VISIBLE_DEVICES="1", HF_TOKEN="hf_secret")
    assert module._cuda_after_import == "", "the adapter process itself must never open a CUDA context"
    post(module, words_wav(4)[0], language="de")
    (process,) = module._fake.processes
    env = process.env
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert "HF_TOKEN" not in env and env["HF_HUB_OFFLINE"] == "1"
    assert env["VLLM_NO_USAGE_STATS"] == "1" and env["VLLM_SERVER_DEV_MODE"] == "1"
    assert module._fake.downloads[0]["token"] == "hf_secret"


def test_every_chunk_of_a_long_recording_is_sent_to_the_engine_at_once(service):
    """Batching is the point: no chunk waits for the one before it."""
    module = service(VLLM_ASR_CHUNK_S="5")
    module._fake.hold = threading.Event()
    wav, levels = words_wav(160, pause_every=8)   # 40 s of words in 2 s groups

    async def scenario():
        async with client(module) as c:
            busy = asyncio.create_task(c.post("/transcribe", data={"language": "de"},
                                              files={"audio": ("a.wav", wav, "audio/wav")}))
            for _ in range(500):
                if module._fake.in_flight >= 8:
                    break
                await asyncio.sleep(0.01)
            seen = module._fake.in_flight
            module._fake.hold.set()
            return seen, await asyncio.wait_for(busy, WAIT)

    seen, r = run(scenario())
    assert r.status_code == 200, r.text
    chunks = r.json()["chunks"]
    assert chunks >= 8 and seen == chunks, f"{seen} of {chunks} chunks were in flight together"
    assert r.json()["text"] == expected_transcript(levels), "joined in the chunks' order"


def test_silence_never_reaches_the_engine(service):
    module = service()
    r = post(module, silent_wav(3.0), language="de")
    assert r.status_code == 200, r.text
    assert r.json()["text"] == "" and r.json()["chunks"] == 0
    assert "No speech" in r.json()["warnings"][0]
    assert module._fake.requests == []


def test_a_chunk_that_stops_on_the_token_cap_is_flagged(service):
    module = service(VLLM_ASR_MAX_NEW_TOKENS="32")
    r = post(module, words_wav(60, pause_every=0)[0], language="de")
    assert r.status_code == 200, r.text
    assert r.json()["truncated"] is True and "VLLM_ASR_MAX_NEW_TOKENS" in r.json()["warnings"][0]


# --- the other families ------------------------------------------------------------------------------------


def test_cohere_is_checked_against_the_digest_and_asked_with_its_language_and_no_prompt(service):
    module = service(**COHERE)
    wav, levels = words_wav(12)
    r = post(module, wav, language="fr", initial_prompt="LUNA")
    assert r.status_code == 200, r.text
    assert r.json()["text"] == expected_transcript(levels)
    assert any("takes no prompt" in w for w in r.json()["warnings"])
    requests = module._fake.requests
    assert {r["path"] for r in requests} == {"/v1/audio/transcriptions"}
    assert all(r["fields"]["language"] == "fr" and "prompt" not in r["fields"] for r in requests)
    assert all(r["fields"]["temperature"] == "0" for r in requests)
    assert module._fake.downloads[0]["revision"] == "29b9036c65620e1a148127c6147543b52358da6a"
    assert call(module, "GET", "/status").json()["weights_check"] == "ok"


def test_the_cohere_mirror_must_carry_the_official_weights_or_the_engine_never_starts(service):
    module = service(VLLM_ASR_MODEL="evewashere/cohere-transcribe-03-2026-ungated")
    r = post(module, words_wav(4)[0], language="de")
    assert r.status_code == 503
    assert module._fake.processes == []
    status = call(module, "GET", "/status").json()
    assert status["weights_check"] == "failed"
    assert call(module, "GET", "/ready").json()["reason"] == "load_failed"


def test_qwen3_identifies_the_language_on_auto_and_takes_the_prompt_as_its_context(service):
    module = service(**QWEN)
    wav, levels = words_wav(8)
    r = post(module, wav, language="auto", initial_prompt="Ein Gespräch über LUNA")
    assert r.status_code == 200, r.text
    assert r.json()["language"] is None and r.json()["warnings"] == []
    assert r.json()["text"] == expected_transcript(levels)
    fields = module._fake.requests[0]["fields"]
    assert "language" not in fields and fields["prompt"] == "Ein Gespräch über LUNA"


def test_no_language_means_the_default_even_for_a_model_that_could_identify_it(service):
    module = service(**QWEN)
    r = post(module, words_wav(4)[0])
    assert r.status_code == 200 and r.json()["language"] == "de"
    assert module._fake.requests[0]["fields"]["language"] == "de"


def test_the_german_whisper_refuses_english_and_reads_auto_as_german_with_a_warning(service):
    module = service(**WHISPER)
    refused = post(module, words_wav(4)[0], language="en")
    assert refused.status_code == 422 and "Supported: de" in refused.json()["detail"]
    assert module._fake.processes == [], "refused before anything was started"
    r = post(module, words_wav(4)[0], language="auto", initial_prompt="Bautzen")
    assert r.status_code == 200 and r.json()["language"] == "de"
    assert "does not identify the language" in r.json()["warnings"][0]
    fields = module._fake.requests[0]["fields"]
    assert fields["language"] == "de" and fields["prompt"] == "Bautzen"


@pytest.mark.parametrize("language", ["it", "German", "xx"])
def test_granite_refuses_a_language_it_does_not_know_before_starting_anything(service, language):
    module = service()
    r = post(module, words_wav(4)[0], language=language)
    assert r.status_code == 422
    assert module._fake.processes == []


def test_a_checkpoint_of_another_kind_is_refused_at_start(service):
    module = service(VLLM_ASR_MODEL="someone/ctc-model", model_type="granite_speech5_ctc")
    assert post(module, words_wav(4)[0], language="de").status_code == 503
    assert module._fake.processes == []


def test_extra_arguments_that_repeat_the_services_own_stop_every_start(service):
    module = service(VLLM_ASR_EXTRA_ARGS="--gpu-memory-utilization 0.9")
    assert post(module, words_wav(4)[0], language="de").status_code == 503
    assert module._fake.processes == []


# --- lifecycle -----------------------------------------------------------------------------------------------


def test_health_ready_and_status_answer_without_starting_the_engine(service):
    module = service()
    health = call(module, "GET", "/health").json()
    assert health["model_resident"] is False and health["engine_state"] == "stopped"
    assert call(module, "GET", "/ready").status_code == 200
    status = call(module, "GET", "/status").json()
    assert status["service"] == "vLLM-ASR" and status["model_family"] == "granite"
    assert status["supported_languages"] == ["de", "en", "es", "fr", "ja", "pt"]
    assert status["engine"]["state"] == "stopped" and status["engine"]["gpu_memory_utilization"] == 0.45
    assert status["engine"]["idle_mode"] == "sleep" and status["queue"]["max_concurrency"] == 32
    assert module._fake.processes == []


def test_unload_puts_the_engine_to_sleep_and_the_next_request_wakes_the_same_process(service):
    module = service()
    post(module, words_wav(4)[0], language="de")
    released = call(module, "POST", "/unload")
    assert released.status_code == 200, released.text
    assert released.json()["engine_state"] == "sleeping" and released.json()["model_resident"] is False
    assert module._fake.sleeping
    again = post(module, words_wav(4)[0], language="de")
    assert again.status_code == 200
    assert len(module._fake.processes) == 1 and "/wake_up" in module._fake.control
    assert call(module, "GET", "/status").json()["engine"]["wake_seconds"] is not None


def test_unload_with_mode_stop_ends_the_process_and_the_next_request_starts_a_new_one(service):
    module = service()
    post(module, words_wav(4)[0], language="de")
    stopped = call(module, "POST", "/unload", params={"mode": "stop"})
    assert stopped.status_code == 200 and stopped.json()["stopped"] is True
    assert stopped.json()["engine_state"] == "stopped"
    assert module._fake.processes[0].returncode is not None
    assert post(module, words_wav(4)[0], language="de").status_code == 200
    assert len(module._fake.processes) == 2
    assert call(module, "POST", "/unload", params={"mode": "nap"}).status_code == 422


def test_idle_mode_stop_ends_the_process_instead_of_sleeping(service):
    module = service(VLLM_ASR_IDLE_MODE="stop")
    post(module, words_wav(4)[0], language="de")
    assert call(module, "POST", "/unload").json()["engine_state"] == "stopped"
    assert "/sleep" not in module._fake.control
    (process,) = module._fake.processes
    assert "--enable-sleep-mode" not in process.argv and "VLLM_SERVER_DEV_MODE" not in process.env


def test_unload_refuses_while_a_transcription_runs(service):
    module = service()
    module._fake.hold = threading.Event()

    async def scenario():
        async with client(module) as c:
            busy = asyncio.create_task(
                c.post("/transcribe", data={"language": "de"}, files={"audio": ("a.wav", words_wav(6)[0])}))
            for _ in range(500):
                if module._fake.entered.is_set():
                    break
                await asyncio.sleep(0.01)
            refused = await c.post("/unload")
            module._fake.hold.set()
            return refused, await asyncio.wait_for(busy, WAIT)

    refused, done = run(scenario())
    assert refused.status_code == 409 and done.status_code == 200


def test_a_start_that_runs_out_of_gpu_memory_is_a_503_and_leaves_no_process(service):
    """#93's lesson for a subprocess: a failed start must give the card back."""
    module = service()
    module._fake.start_log = ["ValueError: ... larger than the available KV cache memory (0.12 GiB)"]
    module._fake.exit_on_start = 1
    r = post(module, words_wav(4)[0], language="de")
    assert r.status_code == 503 and "could not be started" in r.json()["detail"]
    ready = call(module, "GET", "/ready")
    assert ready.status_code == 503 and ready.json()["detail"] == "out_of_memory"
    # The leader exited by itself; its group (the engine core) is still killed.
    assert module._fake.processes and all(p.returncode is not None and p.signals for p in module._fake.processes)
    assert call(module, "GET", "/health").json()["engine_state"] == "stopped"


def test_an_engine_that_dies_mid_request_is_a_503_and_the_next_request_restarts_it(service):
    module = service()
    assert post(module, words_wav(4)[0], language="de").status_code == 200
    module._fake.die_on_chunk = True
    gone = post(module, words_wav(4)[0], language="de")
    assert gone.status_code == 503 and gone.headers.get("retry-after") == "5"
    module._fake.die_on_chunk = False
    assert post(module, words_wav(4)[0], language="de").status_code == 200
    assert len(module._fake.processes) == 2


def test_an_engine_error_answer_is_a_500_with_a_request_id_and_no_detail(service):
    module = service()
    module._fake.chunk_status = 400
    r = post(module, words_wav(4)[0], language="de")
    assert r.status_code == 500 and "Request id" in r.json()["detail"] and "refused" not in r.json()["detail"]


def test_an_empty_upload_and_an_unreadable_one_are_client_errors(service):
    module = service()
    assert post(module, b"", language="de").status_code == 400
    assert post(module, b"this is not audio at all", language="de").status_code == 422
    assert module._fake.processes == []


def test_the_chunk_length_cannot_exceed_the_30_s_every_family_takes_whole(service):
    assert service(VLLM_ASR_CHUNK_S="120").CHUNK_S == 30.0
    assert service().CHUNK_S == 28.0
