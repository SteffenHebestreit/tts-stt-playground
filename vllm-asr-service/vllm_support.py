"""What vllm-asr-service knows about the models it serves (no torch, no vLLM, no network at import).

The service runs ``vllm serve`` as a subprocess and talks to it over HTTP. Which
model, how vLLM must be started for it, and how a chunk is asked of it all depend on
the kind of checkpoint, its *family*, read from the ``model_type`` of its
config.json:

==============  ===================  ===============================================
family          model_type           how a chunk is transcribed
==============  ===================  ===============================================
``whisper``     ``whisper``          ``/v1/audio/transcriptions``; ``prompt`` is
                                     Whisper's initial prompt
``cohere``      ``cohere_asr``       ``/v1/audio/transcriptions``; the language is
                                     part of the decoder prompt, no text prompt
``qwen3-asr``   ``qwen3_asr``        ``/v1/audio/transcriptions``; ``prompt`` is the
                                     context (system turn) Qwen3-ASR biases towards
``granite``     ``granite_speech``   ``/v1/chat/completions`` with the audio as
                                     ``input_audio`` and granite-asr-service's own
                                     prompt (language, punctuation, keywords)
==============  ===================  ===============================================

Why Granite goes through the chat endpoint: vLLM 0.31's transcription prompt for it
is a fixed "can you transcribe the speech into a written format?", with no language
and no keywords, and the 2b answers that in lower case without punctuation (measured
on MLS German clips). The chat endpoint takes the prompt the playground's
granite-asr-service uses, so both services ask the model the same question.

The four checkpoints below were each started under ``vllm/vllm-openai:v0.31.0`` and
measured (vllm-asr-service/README.md); each is pinned to the commit that was
measured. Anything else is served by its family, with the family's languages.
"""

from __future__ import annotations

import base64
import io
import re
import shlex
import wave
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from cohere_support import (
    MIRROR_MODEL as COHERE_MIRROR, MIRROR_REVISION as COHERE_MIRROR_REVISION, OFFICIAL_MODEL as COHERE_OFFICIAL,
    OFFICIAL_WEIGHTS_SHA256 as COHERE_SHA256, OFFICIAL_WEIGHTS_SIZE as COHERE_SIZE,
    SUPPORTED_LANGUAGES as COHERE_LANGUAGES,
)
from granite_support import DEFAULT_MODEL as GRANITE_MODEL, KNOWN_MODELS as GRANITE_KNOWN, build_prompt

FAMILY_WHISPER = "whisper"
FAMILY_COHERE = "cohere"
FAMILY_QWEN3 = "qwen3-asr"
FAMILY_GRANITE = "granite"
FAMILIES = (FAMILY_WHISPER, FAMILY_COHERE, FAMILY_QWEN3, FAMILY_GRANITE)
MODEL_TYPES = {
    "whisper": FAMILY_WHISPER,
    "cohere_asr": FAMILY_COHERE,
    "qwen3_asr": FAMILY_QWEN3,
    "granite_speech": FAMILY_GRANITE,
}

ENDPOINT_TRANSCRIPTIONS = "transcriptions"
ENDPOINT_CHAT = "chat"

# vLLM 0.31's ISO-639-1 list (vllm.model_executor.models.whisper.ISO639_1_SUPPORTED_LANGS):
# what its transcription endpoint accepts as `language` for Whisper and Qwen3-ASR. A
# code outside it is a 400 from vLLM, so the service never sends one.
VLLM_LANGUAGES = frozenset({
    "af", "ar", "az", "be", "bg", "bs", "ca", "cs", "cy", "da", "de", "el", "en", "es", "et", "fa", "fi",
    "fr", "gl", "he", "hi", "hr", "hu", "hy", "id", "is", "it", "ja", "kk", "kn", "ko", "lt", "lv", "mi",
    "mk", "mr", "ms", "ne", "nl", "no", "pl", "pt", "ro", "ru", "sk", "sl", "sr", "sv", "sw", "ta", "th",
    "tl", "tr", "uk", "ur", "vi", "zh",
})
# Qwen3-ASR's own languages (qwen3-asr-service's list) that vLLM's list has too. Not
# Cantonese (`yue` is not in it) and not Filipino (vLLM would name `tl` "Tagalog" in
# the prompt, a name the model was not trained on).
QWEN3_LANGUAGES = frozenset({
    "zh", "en", "ar", "de", "fr", "es", "pt", "id", "it", "ko", "ru", "th", "vi", "ja", "tr", "hi", "ms",
    "nl", "sv", "da", "fi", "pl", "cs", "fa", "el", "ro", "hu", "mk",
})
GRANITE_LANGUAGES = GRANITE_KNOWN[GRANITE_MODEL]["languages"]

LANGUAGES_BY_FAMILY = {
    FAMILY_WHISPER: VLLM_LANGUAGES,
    FAMILY_COHERE: COHERE_LANGUAGES,
    FAMILY_QWEN3: QWEN3_LANGUAGES,
    FAMILY_GRANITE: GRANITE_LANGUAGES,
}
# Families whose model identifies the language when none is given (vLLM asks Whisper
# for its language token; Qwen3-ASR writes "language X<asr_text>" itself, which vLLM
# strips). Cohere and Granite have none: "auto" means the default language there.
DETECTS_LANGUAGE = {FAMILY_WHISPER: True, FAMILY_QWEN3: True, FAMILY_COHERE: False, FAMILY_GRANITE: False}
# What a text prompt (``initial_prompt``/``keywords``) becomes.
PROMPT_USE = {
    FAMILY_WHISPER: "initial prompt",
    FAMILY_QWEN3: "context",
    FAMILY_GRANITE: "keywords",
    FAMILY_COHERE: None,
}

WHISPER_GERMAN = "primeline/whisper-large-v3-turbo-german"
QWEN3_ASR = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_MODEL = GRANITE_MODEL


@dataclass(frozen=True)
class ModelInfo:
    """One checkpoint this service knows before downloading it."""

    family: str
    revision: str
    languages: frozenset
    detects_language: bool
    # Engine arguments the checkpoint needs under vLLM 0.31 (on top of the service's own).
    max_model_len: Optional[int] = None
    sha256: Optional[str] = None
    size: Optional[int] = None
    note: str = ""


# The measured checkpoints. max_model_len: vLLM sizes the KV cache for one request of
# the model's full context at start-up; Qwen3-ASR's 65536 and granite-4.0-1b's 16384
# do not fit in the VRAM share this service gets, and a 30 s chunk needs a few hundred
# tokens (Granite ~10 audio tokens/s, Qwen3-ASR ~13/s, plus the transcript).
KNOWN_MODELS: Mapping[str, ModelInfo] = {
    GRANITE_MODEL: ModelInfo(
        family=FAMILY_GRANITE, revision=GRANITE_KNOWN[GRANITE_MODEL]["revision"], languages=GRANITE_LANGUAGES,
        detects_language=False, max_model_len=2048,
        note="IBM Granite Speech 5.0 2b; prompted with the language, punctuation and keywords"),
    COHERE_MIRROR: ModelInfo(
        family=FAMILY_COHERE, revision=COHERE_MIRROR_REVISION, languages=COHERE_LANGUAGES, detects_language=False,
        sha256=COHERE_SHA256, size=COHERE_SIZE,
        note="Cohere Transcribe 03-2026 from the ungated mirror, weights checked against Cohere's digest"),
    COHERE_OFFICIAL: ModelInfo(
        family=FAMILY_COHERE, revision="main", languages=COHERE_LANGUAGES, detects_language=False,
        note="Cohere Transcribe 03-2026, the gated official repository (HF_TOKEN)"),
    QWEN3_ASR: ModelInfo(
        family=FAMILY_QWEN3, revision="7278e1e70fe206f11671096ffdd38061171dd6e5", languages=QWEN3_LANGUAGES,
        detects_language=True, max_model_len=2048,
        note="Qwen3-ASR 1.7B; identifies the language, takes a context prompt"),
    WHISPER_GERMAN: ModelInfo(
        family=FAMILY_WHISPER, revision="9e7012dad9247f07553aac8a99c4d50ad7c3d36d", languages=frozenset({"de"}),
        detects_language=False,
        note="Whisper large-v3-turbo fine-tuned on German (primeline); German only"),
}
# Shown in /status and the README; the TrueNAS form offers exactly these.
OFFERED_MODELS = (GRANITE_MODEL, COHERE_MIRROR, QWEN3_ASR, WHISPER_GERMAN)

# The longest chunk every family takes whole: Whisper's and Qwen3-ASR's feature
# extractors work on 30 s windows, and vLLM would split anything longer itself (with
# its own seams, which the service's pause-based cuts exist to avoid).
MAX_CHUNK_S = 30.0
DEFAULT_CHUNK_S = 28.0


class ModelChoiceError(ValueError):
    """A configured checkpoint this service cannot run; ``str()`` names what it can."""


def family_for(model_type: Optional[str]) -> str:
    """The family a config.json ``model_type`` belongs to; ModelChoiceError for one vLLM is not asked to serve here."""
    family = MODEL_TYPES.get((model_type or "").strip())
    if family is None:
        raise ModelChoiceError(
            f"model_type {model_type!r} is not a speech-to-text model this service runs "
            f"(supported: {', '.join(sorted(MODEL_TYPES))})")
    return family


def default_revision(model: str) -> str:
    """The pinned commit of a measured checkpoint, ``main`` for anything else."""
    known = KNOWN_MODELS.get(model)
    return known.revision if known else "main"


def default_sha256(model: str) -> Optional[str]:
    """The digest model.safetensors must have when VLLM_ASR_SHA256 says nothing (only the Cohere mirror has one)."""
    known = KNOWN_MODELS.get(model)
    return known.sha256 if known else None


def parse_language_list(raw: Optional[str]) -> frozenset:
    """``VLLM_ASR_SUPPORTED_LANGUAGES``: comma/space separated codes, locale tags reduced; empty for none."""
    codes = {part.strip().lower().replace("_", "-").split("-", 1)[0] for part in re.split(r"[,\s;]+", raw or "")}
    return frozenset(code for code in codes if code.isalpha() and 2 <= len(code) <= 3)


def supported_languages(model: str, family: str, override: Optional[str] = None) -> frozenset:
    """Languages to accept: the override when it names any, else the model's own list, else its family's.

    Never more than vLLM's own list for the families whose language goes to vLLM as a
    code (Whisper, Qwen3-ASR): anything else would be a 400 from the engine.
    """
    chosen = parse_language_list(override)
    if not chosen:
        known = KNOWN_MODELS.get(model)
        chosen = known.languages if known else LANGUAGES_BY_FAMILY[family]
    if family in (FAMILY_WHISPER, FAMILY_QWEN3):
        chosen = frozenset(chosen & VLLM_LANGUAGES)
    return chosen


def detects_language(model: str, family: str) -> bool:
    known = KNOWN_MODELS.get(model)
    return known.detects_language if known else DETECTS_LANGUAGE[family]


def endpoint_for(family: str) -> str:
    return ENDPOINT_CHAT if family == FAMILY_GRANITE else ENDPOINT_TRANSCRIPTIONS


# --- the engine's command line ---------------------------------------------------------------


@dataclass(frozen=True)
class EngineSettings:
    """How ``vllm serve`` is started (every field is a VLLM_ASR_* setting)."""

    gpu_memory_utilization: float = 0.45
    max_model_len: Optional[int] = None
    max_num_seqs: Optional[int] = None
    max_num_batched_tokens: Optional[int] = None
    sleep_mode: bool = True
    extra_args: Sequence[str] = field(default_factory=tuple)


def parse_extra_args(raw: Optional[str]) -> tuple[str, ...]:
    """``VLLM_ASR_EXTRA_ARGS`` split like a shell would (``--enforce-eager --dtype float16``)."""
    return tuple(shlex.split(raw or ""))


# Arguments the service sets itself: an extra argument that repeats one would leave
# two values on the command line, and vLLM takes the last, so it is refused instead.
OWN_ARGS = frozenset({
    "--host", "--port", "--served-model-name", "--gpu-memory-utilization", "--max-model-len",
    "--max-num-seqs", "--max-num-batched-tokens", "--enable-sleep-mode", "--revision", "--model",
    "--trust-remote-code", "--api-key",
})


def check_extra_args(args: Iterable[str]) -> list[str]:
    """The extra arguments that collide with the service's own, ``--x=1`` and ``--x 1`` alike."""
    return [arg for arg in args if arg.split("=", 1)[0] in OWN_ARGS]


def engine_argv(model_dir: str, served_name: str, port: int, settings: EngineSettings,
                default_max_model_len: Optional[int] = None, executable: Sequence[str] = ("vllm", "serve")) -> list[str]:
    """The ``vllm serve`` command line for *model_dir*, listening on 127.0.0.1:*port* only.

    No remote code is ever trusted: the four families are vLLM's own model classes.
    """
    argv = [*executable, model_dir, "--served-model-name", served_name, "--host", "127.0.0.1",
            "--port", str(int(port)), "--gpu-memory-utilization", f"{settings.gpu_memory_utilization:g}"]
    max_model_len = settings.max_model_len or default_max_model_len
    if max_model_len:
        argv += ["--max-model-len", str(int(max_model_len))]
    if settings.max_num_seqs:
        argv += ["--max-num-seqs", str(int(settings.max_num_seqs))]
    if settings.max_num_batched_tokens:
        argv += ["--max-num-batched-tokens", str(int(settings.max_num_batched_tokens))]
    if settings.sleep_mode:
        argv.append("--enable-sleep-mode")
    argv += list(settings.extra_args)
    return argv


# --- one chunk as a request -------------------------------------------------------------------


def wav_bytes(samples: np.ndarray, rate: int = 16000) -> bytes:
    """16-bit mono PCM WAV of *samples* (float, -1..1): what the engine decodes for a chunk.

    Scaled by 32768 and clipped, the inverse of what every decoder does (int16 / 32768),
    so audio that arrived as 16-bit PCM reaches the engine bit for bit. Half the size
    of float WAV; the request never leaves the container (127.0.0.1).
    """
    scaled = np.rint(np.asarray(samples, dtype=np.float64) * 32768.0)
    pcm = np.clip(scaled, -32768, 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm.tobytes())
    return buffer.getvalue()


def prompt_text(family: str, initial_prompt: Optional[str], keywords: Sequence[str]) -> str:
    """The text prompt a family gets from ``initial_prompt`` and the keyword list ("" for none).

    Granite's keywords go into its own prompt (``chat_body``), so nothing here.
    """
    if family in (FAMILY_COHERE, FAMILY_GRANITE):
        return ""
    prompt = (initial_prompt or "").strip()
    words = [w for w in keywords if w]
    if words and not prompt:
        prompt = ", ".join(words)
    return prompt


def transcription_fields(model: str, language: Optional[str], prompt: str, max_tokens: int) -> dict:
    """The form fields of one ``/v1/audio/transcriptions`` call (the file goes alongside).

    Greedy (temperature 0), like every other engine here. *language* None asks the
    model to identify it (Whisper, Qwen3-ASR).
    """
    fields = {"model": model, "response_format": "json", "temperature": "0",
              "max_completion_tokens": str(int(max_tokens))}
    if language:
        fields["language"] = language
    if prompt:
        fields["prompt"] = prompt
    return fields


def chat_body(model: str, audio_wav: bytes, language: Optional[str], keywords: Sequence[str],
              max_tokens: int) -> dict:
    """The ``/v1/chat/completions`` body for one Granite chunk: the audio first, then the prompt.

    vLLM puts the model's audio placeholder where the audio part is, so the prompt is
    granite-asr-service's without its leading ``<|audio|>``.
    """
    text = build_prompt(language, keywords)
    if text.startswith("<|audio|>"):
        text = text[len("<|audio|>"):]
    audio = base64.b64encode(audio_wav).decode("ascii")
    return {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": audio, "format": "wav"}},
            {"type": "text", "text": text},
        ]}],
        "temperature": 0,
        "max_tokens": int(max_tokens),
    }


@dataclass(frozen=True)
class EngineText:
    text: str
    capped: bool = False


def parse_transcription(payload) -> EngineText:
    """The text of a ``/v1/audio/transcriptions`` json answer, without Whisper's leading space."""
    if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
        raise ValueError("the engine's transcription answer has no text")
    return EngineText(text=payload["text"].strip())


def parse_chat(payload) -> EngineText:
    """The text of a ``/v1/chat/completions`` answer, and whether it stopped on the token cap."""
    try:
        choice = payload["choices"][0]
        text = choice["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ValueError("the engine's chat answer has no message") from None
    if not isinstance(text, str):
        text = ""
    return EngineText(text=text.strip(), capped=choice.get("finish_reason") == "length")
