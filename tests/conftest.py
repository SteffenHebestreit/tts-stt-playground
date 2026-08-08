"""Shared fixtures for service integration tests."""

import os
import struct
import math

import pytest
import httpx

# Tests in this suite come in two kinds:
#
#   * unit tests, which import service modules straight off the filesystem and
#     run anywhere;
#   * live tests, which drive a running container over HTTP.
#
# The live ones resolve Docker-network hostnames like `stt-service`, so outside
# the stack they used to produce ~33 identical `getaddrinfo failed` errors. That
# is noise, not signal: it buries a genuine regression in a wall of red and
# makes the suite useless as a pre-commit check.
#
# So a live fixture probes its service once per session and skips if it is not
# there. Inside the compose stack the services *must* be up, and skipping would
# hide real breakage -- set REQUIRE_LIVE_SERVICES=1 there to turn the skip back
# into a hard failure (docker-compose.yml does this for the `tests` service).
REQUIRE_LIVE_SERVICES = os.getenv("REQUIRE_LIVE_SERVICES", "").lower() in (
    "1",
    "true",
    "yes",
)

_reachability_cache: dict[str, bool] = {}


def _service_is_up(base_url: str) -> bool:
    """One cached probe per base URL, so N fixtures cost at most N connects."""
    if base_url not in _reachability_cache:
        try:
            # Any response at all proves the service is listening; a 404 on /
            # is still a live service, so the status code is irrelevant here.
            httpx.get(base_url, timeout=httpx.Timeout(3.0, connect=2.0))
            _reachability_cache[base_url] = True
        except httpx.HTTPError:
            _reachability_cache[base_url] = False
    return _reachability_cache[base_url]


def live_client(base_url: str, timeout: float) -> httpx.Client:
    """Client for a live service, or skip the tests that need it."""
    if not REQUIRE_LIVE_SERVICES and not _service_is_up(base_url):
        pytest.skip(
            f"live service not reachable at {base_url} — start the stack with "
            f"`docker compose up -d` to run these, or set "
            f"REQUIRE_LIVE_SERVICES=1 to make this a failure instead"
        )
    return httpx.Client(base_url=base_url, timeout=timeout)


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
    """HTTP client for the Piper TTS service."""
    with live_client(PIPER_TTS_URL, 30.0) as client:
        yield client


@pytest.fixture(scope="session")
def stt_client():
    """HTTP client for the STT service."""
    with live_client(STT_URL, 60.0) as client:
        yield client


@pytest.fixture(scope="session")
def training_client():
    """HTTP client for the training service."""
    with live_client(TRAINING_URL, 30.0) as client:
        yield client


@pytest.fixture(scope="session")
def qwen3_tts_client():
    """HTTP client for the Qwen3 TTS service."""
    with live_client(QWEN3_TTS_URL, 120.0) as client:
        yield client


@pytest.fixture(scope="session")
def qwen3_asr_client():
    """HTTP client for the Qwen3 ASR service."""
    with live_client(QWEN3_ASR_URL, 60.0) as client:
        yield client


@pytest.fixture(scope="session")
def frontend_client():
    """HTTP client for the frontend service."""
    with live_client(FRONTEND_URL, 10.0) as client:
        yield client
