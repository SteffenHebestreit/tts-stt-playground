"""Shared fixtures for service integration tests."""

import os
import struct
import math

import pytest
import httpx

# Service URLs from environment or defaults
PIPER_TTS_URL = os.getenv("PIPER_TTS_URL", "http://piper-tts-service:5000")
STT_URL = os.getenv("STT_URL", "http://stt-service:8000")
TRAINING_URL = os.getenv("TRAINING_URL", "http://piper-training-service:8080")
QWEN3_TTS_URL = os.getenv("QWEN3_TTS_URL", "http://qwen3-tts-service:5004")
QWEN3_ASR_URL = os.getenv("QWEN3_ASR_URL", "http://qwen3-asr-service:5002")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://frontend-service:3000")


def generate_sine_wav(duration_s: float = 2.0, sample_rate: int = 22050, freq: float = 440.0) -> bytes:
    """Generate a short sine-wave WAV file in memory."""
    num_samples = int(sample_rate * duration_s)
    samples = []
    for i in range(num_samples):
        value = int(32767 * math.sin(2 * math.pi * freq * i / sample_rate))
        samples.append(struct.pack("<h", value))
    raw = b"".join(samples)
    # Build WAV header
    data_size = len(raw)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,        # chunk size
        1,         # PCM
        1,         # mono
        sample_rate,
        sample_rate * 2,  # byte rate
        2,         # block align
        16,        # bits per sample
        b"data",
        data_size,
    )
    return header + raw


@pytest.fixture(scope="session")
def test_audio_bytes() -> bytes:
    """A short 2-second 440 Hz sine wave WAV."""
    return generate_sine_wav(duration_s=2.0)


@pytest.fixture(scope="session")
def piper_tts_client():
    return httpx.Client(base_url=PIPER_TTS_URL, timeout=30.0)


@pytest.fixture(scope="session")
def stt_client():
    return httpx.Client(base_url=STT_URL, timeout=60.0)


@pytest.fixture(scope="session")
def training_client():
    return httpx.Client(base_url=TRAINING_URL, timeout=30.0)


@pytest.fixture(scope="session")
def qwen3_tts_client():
    return httpx.Client(base_url=QWEN3_TTS_URL, timeout=120.0)


@pytest.fixture(scope="session")
def qwen3_asr_client():
    return httpx.Client(base_url=QWEN3_ASR_URL, timeout=60.0)


@pytest.fixture(scope="session")
def frontend_client():
    return httpx.Client(base_url=FRONTEND_URL, timeout=10.0)
