"""Cohere Transcribe specifics for cohere-asr-service (no torch or transformers at import).

Where the weights come from, and why from a mirror by default:

``CohereLabs/cohere-transcribe-03-2026`` is Apache-2.0 but gated on Hugging Face: an
anonymous download answers 401, so every install would need a personal access token
first. ``evewashere/cohere-transcribe-03-2026-ungated`` republishes the same files
without the gate. The default pins that mirror to one commit, in which every file
has the same git blob id as in the official repository except README.md, and
``model.safetensors`` has the official LFS SHA-256 and size below. The service
hashes the weights before loading them and refuses to load anything else
(``COHERE_ASR_SHA256``), so a mirror that changed its files could not change what
runs here. With a token (``HF_TOKEN``) the official repository can be used instead:
``COHERE_ASR_MODEL=CohereLabs/cohere-transcribe-03-2026``.
"""

from __future__ import annotations

from typing import Optional

MIRROR_MODEL = "evewashere/cohere-transcribe-03-2026-ungated"
MIRROR_REVISION = "29b9036c65620e1a148127c6147543b52358da6a"
OFFICIAL_MODEL = "CohereLabs/cohere-transcribe-03-2026"
# The LFS pointer of model.safetensors in the official repository.
OFFICIAL_WEIGHTS_SHA256 = "987bd3e141c7bfdb5a78f5db11397ee7737308357e6cc0a3f36a4979b158137a"
OFFICIAL_WEIGHTS_SIZE = 4131862976

# The 14 languages of the model card (and of transformers' CohereAsrProcessor, which
# raises ValueError for anything else). There is no language identification: the
# language is part of the decoder prompt.
SUPPORTED_LANGUAGES = frozenset({
    "ar", "de", "el", "en", "es", "fr", "it", "ja", "ko", "nl", "pl", "pt", "vi", "zh",
})

# The feature extractor splits anything longer than max_audio_clip_s minus
# overlap_chunk_second (35 - 5 s) itself, at its own energy minima, and returns the
# pieces' text joined; the service cuts first so that every chunk, and so every
# segment, keeps the start and end it was cut at.
MAX_CHUNK_S = 30.0


def default_revision(model: str) -> str:
    """The pinned commit for the default mirror; ``main`` for anything else."""
    return MIRROR_REVISION if model == MIRROR_MODEL else "main"


def default_sha256(model: str) -> Optional[str]:
    """The digest the weights must have when ``COHERE_ASR_SHA256`` says nothing.

    Only the mirror is held to the official digest by default: the official
    repository is the source of truth itself, and a local directory or another
    repository has no known digest (set ``COHERE_ASR_SHA256`` for those).
    """
    return OFFICIAL_WEIGHTS_SHA256 if model == MIRROR_MODEL else None


def expected_size(sha256: Optional[str]) -> Optional[int]:
    """The official file size, when the official digest is the one expected."""
    return OFFICIAL_WEIGHTS_SIZE if sha256 == OFFICIAL_WEIGHTS_SHA256 else None


def generated_lengths(sequences, prompt_length: int, pad_token_id: Optional[int],
                      eos_token_id: Optional[int]) -> list[int]:
    """New tokens per row of ``generate()``'s output (prompt, padding and end token not counted).

    A row whose count reaches ``max_new_tokens`` stopped on the cap, not on the end
    token: its text is probably cut off.
    """
    lengths = []
    for row in sequences:
        tokens = [int(t) for t in list(row)[prompt_length:]]
        count = 0
        for token in tokens:
            if token == eos_token_id or token == pad_token_id:
                break
            count += 1
        lengths.append(count)
    return lengths
