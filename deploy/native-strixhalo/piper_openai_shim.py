#!/usr/bin/env python3
"""
OpenAI Speech API for Piper TTS.

Why this exists: Piper is a library and a CLI, not an OpenAI endpoint. This
shim loads the voices once, keeps them resident and speaks the documented
OpenAI Speech API, so llama-swap can supervise it like any other model and
clients reach it with the same keys and the same base URL as the LLMs.

THE ONLY PUBLIC SURFACE IS THE OPENAI API:

  POST /v1/audio/speech   application/json
       input (required, <= 4096 chars), model, voice, response_format, speed
  GET  /v1/models         the installed voices
  GET  /health            503 until every voice is loaded (llama-swap gate)
  GET  /                  the same, for llama-swap's preload probe

voice: an installed Piper voice id (de_DE-thorsten-high), a language or locale
(de, de_DE, en_US) for that language's first installed voice, or one of
OpenAI's own voice names (alloy, onyx, ...), which mean nothing to Piper and
select the default voice. Clients hardcode those names, so refusing them would
break every stock OpenAI client.

response_format: mp3 (the spec's default), wav, pcm (raw 16-bit little-endian
mono at 24 kHz, as the spec defines it). opus, aac and flac are refused with
code unsupported_value: there is no encoder for them on this host.

Every other documented OpenAI parameter this backend cannot honour
(instructions, stream_format=sse) is REJECTED with its name in `param`, never
accepted and ignored.
"""

import argparse
import io
import json
import signal
import sys
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from piper import PiperVoice, SynthesisConfig

MAX_INPUT_CHARS = 4096  # the OpenAI limit
PCM_RATE = 24000  # the spec's pcm format
FORMATS = {"mp3": "audio/mpeg", "wav": "audio/wav", "pcm": "audio/pcm"}
KNOWN_UNSUPPORTED_FORMATS = {"opus", "aac", "flac"}
OPENAI_VOICES = {
    "alloy", "echo", "fable", "onyx", "nova", "shimmer",
    "ash", "ballad", "coral", "sage", "verse", "marin", "cedar",
}

VOICES = {}  # id -> {"path", "language"}; filled at start from --voices
LOADED = {}  # id -> PiperVoice
LOCKS = {}  # id -> Lock: one synthesis per voice at a time
LOAD_LOCK = threading.Lock()
ARGS = None
READY = threading.Event()


def scan_voices(folder):
    for model in sorted(Path(folder).glob("*.onnx")):
        config = model.with_suffix(".onnx.json")
        if not config.exists():
            continue
        lang = json.loads(config.read_text(encoding="utf-8")).get("language", {}).get("code", "")
        VOICES[model.name[: -len(".onnx")]] = {"path": model, "language": lang}
        LOCKS[model.name[: -len(".onnx")]] = threading.Lock()


def voice(vid):
    with LOAD_LOCK:
        if vid not in LOADED:
            t0 = time.time()
            LOADED[vid] = PiperVoice.load(VOICES[vid]["path"])
            log(f"loaded {vid} in {time.time() - t0:.1f}s")
        return LOADED[vid]


def resolve_voice(requested):
    """(voice id, None) or (None, error message)."""
    v = (requested or "").strip()
    if not v or v.lower() in OPENAI_VOICES:
        return ARGS.default_voice, None
    if v in VOICES:
        return v, None
    want = v.replace("-", "_").lower()
    for vid, meta in VOICES.items():  # a language or locale: de, de_DE, en_us
        code = meta["language"].lower()
        if want in (code, code.split("_")[0]):
            return vid, None
    return None, f"unknown voice {v!r}; installed: {', '.join(VOICES)}"


def synthesize(vid, text, speed):
    """16-bit mono PCM samples and their rate."""
    v = voice(vid)
    buf = io.BytesIO()
    with LOCKS[vid], wave.open(buf, "wb") as w:
        v.synthesize_wav(text, w, syn_config=SynthesisConfig(length_scale=1.0 / speed))
    buf.seek(0)
    with wave.open(buf, "rb") as r:
        return r.readframes(r.getnframes()), r.getframerate()


def encode(pcm, rate, fmt):
    if fmt == "wav":
        out = io.BytesIO()
        with wave.open(out, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(pcm)
        return out.getvalue()
    if fmt == "pcm":
        if rate == PCM_RATE:
            return pcm
        import soxr

        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        resampled = soxr.resample(samples, rate, PCM_RATE)
        return (np.clip(resampled, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    import lameenc  # mp3

    enc = lameenc.Encoder()
    enc.set_bit_rate(64)
    enc.set_in_sample_rate(rate)
    enc.set_channels(1)
    enc.set_quality(2)
    return enc.encode(pcm) + enc.flush()


def log(msg):
    print(f"[piper-shim] {msg}", file=sys.stderr, flush=True)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):
        pass

    def _send(self, code, body, content_type="application/json"):
        if isinstance(body, dict):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code, msg, typ="invalid_request_error", param=None, err_code=None):
        self._send(code, {"error": {"message": msg, "type": typ, "param": param, "code": err_code}})

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/":  # llama-swap's preload probes the root
            self._send(200 if READY.is_set() else 503, {"service": "piper", "ready": READY.is_set()})
        elif path == "/health":
            if READY.is_set():
                self._send(200, {"status": "ok", "voices": list(VOICES), "default_voice": ARGS.default_voice})
            else:
                self._send(503, {"status": "loading"})
        elif path == "/v1/models":
            data = [{"id": vid, "object": "model", "owned_by": "piper", "language": m["language"]}
                    for vid, m in VOICES.items()]
            self._send(200, {"object": "list", "data": data})
        else:
            self._error(404, f"no route {path}", err_code="not_found")

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path != "/v1/audio/speech":
            self._error(404, f"no route {path}", err_code="not_found")
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("not an object")
        except (ValueError, json.JSONDecodeError):
            self._error(400, "Request body must be a JSON object.")
            return

        text = body.get("input")
        if not isinstance(text, str) or not text.strip():
            self._error(400, "input is required", param="input")
            return
        if len(text) > MAX_INPUT_CHARS:
            self._error(400, f"input is longer than {MAX_INPUT_CHARS} characters", param="input")
            return
        if body.get("instructions"):
            self._error(400, "Piper voices cannot follow instructions", param="instructions",
                        err_code="unsupported_parameter")
            return
        if body.get("stream_format") not in (None, "audio"):
            self._error(400, "only stream_format=audio is supported", param="stream_format",
                        err_code="unsupported_value")
            return
        fmt = (body.get("response_format") or "mp3").lower()
        if fmt not in FORMATS:
            known = fmt in KNOWN_UNSUPPORTED_FORMATS
            self._error(400, f"response_format {fmt!r} is not supported here; use mp3, wav or pcm",
                        param="response_format", err_code="unsupported_value" if known else "invalid_value")
            return
        try:
            speed = float(body.get("speed", 1.0))
        except (TypeError, ValueError):
            speed = -1.0
        if not 0.25 <= speed <= 4.0:
            self._error(400, "speed must be between 0.25 and 4.0", param="speed")
            return
        vid, problem = resolve_voice(body.get("voice"))
        if problem:
            self._error(400, problem, param="voice", err_code="invalid_value")
            return

        t0 = time.time()
        try:
            pcm, rate = synthesize(vid, text, speed)
            audio = encode(pcm, rate, fmt)
        except Exception as e:  # noqa: BLE001 — a failed synthesis is a server error, not a crash
            log(f"synthesis failed ({vid}): {e!r}")
            self._error(500, "speech synthesis failed", typ="server_error")
            return
        secs = len(pcm) / 2 / rate
        log(f"{vid} {fmt} {len(text)} chars -> {secs:.1f}s audio in {time.time() - t0:.2f}s")
        self._send(200, audio, FORMATS[fmt])


def main():
    global ARGS
    p = argparse.ArgumentParser(description="OpenAI Speech API for Piper TTS")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--listen-port", type=int, required=True)
    p.add_argument("--voices", required=True, help="folder with <voice>.onnx + <voice>.onnx.json")
    p.add_argument("--default-voice", required=True)
    ARGS = p.parse_args()

    scan_voices(ARGS.voices)
    if ARGS.default_voice not in VOICES:
        sys.exit(f"default voice {ARGS.default_voice!r} not in {ARGS.voices} ({', '.join(VOICES) or 'empty'})")
    server = ThreadingHTTPServer((ARGS.host, ARGS.listen_port), Handler)
    server.daemon_threads = True
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown).start())

    def warm():
        for vid in VOICES:  # load every voice, so the first request is not the slow one
            synthesize(vid, "Hallo." if VOICES[vid]["language"].startswith("de") else "Hello.", 1.0)
        READY.set()
        log(f"ready: {', '.join(VOICES)} (default {ARGS.default_voice})")

    threading.Thread(target=warm, daemon=True).start()
    log(f"listening on {ARGS.host}:{ARGS.listen_port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
