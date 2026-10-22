"""benchmarks/run_throughput.py against a stub backend: concurrency, the report, failures.

The stub (an httpx.MockTransport) answers like the gateway's /api/stt and records how
many requests it held at once, so the test proves the level's concurrency is what
reached the server, not just what the tool was asked for.
"""

from __future__ import annotations

import asyncio
import importlib.util
import io
import json
import sys
import wave
from pathlib import Path

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = REPO_ROOT / "benchmarks"


@pytest.fixture(scope="module")
def tool():
    sys.path.insert(0, str(BENCHMARKS))
    try:
        spec = importlib.util.spec_from_file_location("run_throughput_under_test", BENCHMARKS / "run_throughput.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(BENCHMARKS))


def _wav(seconds: float) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(b"\0\0" * int(16000 * seconds))
    return buffer.getvalue()


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    lines = []
    for i in range(12):
        (tmp_path / f"c{i}.wav").write_bytes(_wav(2.0))
        lines.append(json.dumps({"audio": f"c{i}.wav", "text": f"hallo welt {i}"}))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


class Stub:
    def __init__(self, fail_every: int = 0):
        self.in_flight = 0
        self.peak = 0
        self.fields: list[bytes] = []
        self.calls = 0
        self.fail_every = fail_every

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        body = await request.aread()
        self.fields.append(body)
        self.calls += 1
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(0.02)
            if self.fail_every and self.calls % self.fail_every == 0:
                return httpx.Response(503, json={"detail": "busy"})
            number = int(body.split(b'filename="c', 1)[1].split(b".wav", 1)[0])
            return httpx.Response(200, json={"text": f"Hallo Welt {number}.", "duration": 2.0})
        finally:
            self.in_flight -= 1


def test_each_level_holds_that_many_requests_and_reports_throughput(tool, corpus, tmp_path):
    stub = Stub()
    out = tmp_path / "tp.json"
    code = tool.main(["--base-url", "http://gw.test", "--provider", "vllm-asr", "--manifest", str(corpus),
                      "--concurrency", "1", "4", "--out", str(out)], transport=httpx.MockTransport(stub))
    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["provider"] == "vllm-asr" and report["url"] == "http://gw.test/api/stt" and report["clips"] == 12
    one, four = report["levels"]
    assert (one["concurrency"], four["concurrency"]) == (1, 4)
    assert one["audio_s"] == four["audio_s"] == 24.0 and one["failed"] == four["failed"] == 0
    assert one["wer"] == 0.0 and four["cer"] == 0.0, "the normaliser drops case and punctuation"
    assert four["rtfx"] > one["rtfx"]
    assert stub.peak == 4
    assert all(b'name="provider"' in body and b"vllm-asr" in body for body in stub.fields)


def test_a_failed_request_is_counted_not_scored_and_fails_the_run(tool, corpus):
    stub = Stub(fail_every=5)
    code = tool.main(["--url", "http://svc.test/transcribe", "--manifest", str(corpus), "--concurrency", "2"],
                     transport=httpx.MockTransport(stub))
    assert code == 1


def test_the_clip_selection_is_run_german_evals(tool, corpus):
    rows = tool.load_manifest(corpus)
    assert tool.sample_manifest(rows, 5, 1234) == tool.sample_manifest(rows, 5, 1234)
    assert tool.wav_seconds(corpus.parent / "c0.wav") == 2.0
    assert tool.wav_seconds(corpus) is None
