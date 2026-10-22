#!/usr/bin/env python3
"""Measure STT throughput under concurrency: audio seconds transcribed per wall-clock second.

run_german_eval.py sends one clip at a time, which measures accuracy and latency but
not what a batching engine (vllm-asr) is for. This sends the same clips with N
requests in flight at once, for each N given, and reports per level:

    rtfx      audio seconds / wall seconds (the throughput; 100 = 100x real time)
    req/s     finished requests per second
    p50, p95  request latency, which grows with N once the engine is saturated
    WER, CER  of what came back (asr_metrics' normaliser), so a fast but broken
              configuration does not look good

    # one provider through the gateway, 1, 8 and 32 requests at once
    python benchmarks/run_throughput.py --base-url http://localhost:3000 --provider vllm-asr \\
        --manifest data/manifest.jsonl --limit 200 --concurrency 1 8 32 --out tp.json

    # a backend directly
    python benchmarks/run_throughput.py --url http://localhost:5011/transcribe --manifest data/de.tsv

The clip selection (--limit, --seed) is run_german_eval's, so both tools can measure
the same audio. Each level starts with one untimed warm-up request. Through the
gateway, MAX_CONCURRENT_UPLOADS (per gateway worker, 4 by default) refuses requests
beyond it with 503: raise it to at least the highest level, or measure the backend
directly. A refused or failed request is counted and printed, never scored.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import wave
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

for _stream in (sys.stdout, sys.stderr):  # German output on a Windows console (see run_german_eval)
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from asr_metrics import corpus_rate  # noqa: E402
from run_german_eval import (  # noqa: E402
    API_CHOICES, DEFAULT_BASE_URL, DEFAULT_SEED, Target, _extract_text, _request_fields, load_manifest,
    resolve_target, sample_manifest,
)


def wav_seconds(path: Path) -> Optional[float]:
    """A WAV file's length from its header, or None for anything else."""
    try:
        with wave.open(str(path), "rb") as handle:
            return handle.getnframes() / float(handle.getframerate())
    except (wave.Error, EOFError, OSError):
        return None


async def _one(client, target: Target, audio: bytes, name: str, *, language: str, model: str,
               headers: Optional[dict]) -> tuple[float, dict]:
    field, data = _request_fields(target, model=model, language=language)
    started = time.perf_counter()
    response = await client.post(target.url, files={field: (name, audio, "application/octet-stream")},
                                 data=data, headers=headers)
    elapsed = time.perf_counter() - started
    if response.status_code != 200:
        raise RuntimeError(f"{name}: HTTP {response.status_code} - {response.text[:160]}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"{name}: unexpected response shape {type(payload).__name__}")
    return elapsed, payload


async def run_level(client, target: Target, clips: list[tuple[str, bytes, str, Optional[float]]],
                    concurrency: int, *, language: str = "de", model: str = "whisper-1",
                    headers: Optional[dict] = None) -> dict:
    """All *clips* with *concurrency* requests in flight; the level's report."""
    name, audio, _, _ = clips[0]
    await _one(client, target, audio, name, language=language, model=model, headers=headers)  # warm-up
    gate = asyncio.Semaphore(concurrency)
    results: dict[int, tuple[float, dict]] = {}
    failures: list[str] = []

    async def go(index: int) -> None:
        clip_name, clip_audio, _, _ = clips[index]
        async with gate:
            try:
                results[index] = await _one(client, target, clip_audio, clip_name, language=language,
                                            model=model, headers=headers)
            except Exception as exc:  # counted, printed, not scored
                failures.append(str(exc))

    started = time.perf_counter()
    await asyncio.gather(*(go(i) for i in range(len(clips))))
    wall = time.perf_counter() - started

    audio_s = 0.0
    pairs = []
    latencies = []
    for index, (elapsed, payload) in results.items():
        _, _, reference, header_seconds = clips[index]
        stated = payload.get("duration")
        seconds = float(stated) if isinstance(stated, (int, float)) and stated > 0 else (header_seconds or 0.0)
        audio_s += seconds
        latencies.append(elapsed)
        pairs.append((reference, _extract_text(payload)))
    latencies.sort()
    return {
        "concurrency": concurrency,
        "requests": len(clips),
        "failed": len(failures),
        "audio_s": round(audio_s, 1),
        "wall_s": round(wall, 2),
        "rtfx": round(audio_s / wall, 1) if wall > 0 else None,
        "req_per_s": round(len(results) / wall, 2) if wall > 0 else None,
        "lat_p50": round(statistics.median(latencies), 3) if latencies else None,
        "lat_p95": round(latencies[int(0.95 * (len(latencies) - 1))], 3) if latencies else None,
        "wer": round(100 * corpus_rate(pairs).rate, 2) if pairs else None,
        "cer": round(100 * corpus_rate(pairs, character_level=True).rate, 2) if pairs else None,
        "first_failures": failures[:3],
    }


async def _run(args: argparse.Namespace, transport=None) -> dict:
    import httpx

    target = resolve_target(args.base_url, args.url, args.api, args.provider)
    rows = sample_manifest(load_manifest(Path(args.manifest), Path(args.audio_root) if args.audio_root else None),
                           args.limit, args.seed)
    clips = [(path.name, path.read_bytes(), text, wav_seconds(path)) for path, text in rows]
    if not clips:
        raise SystemExit("the manifest selected no clips")
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else None
    levels = []
    for concurrency in args.concurrency:
        limits = httpx.Limits(max_connections=concurrency + 4, max_keepalive_connections=concurrency + 4)
        async with httpx.AsyncClient(timeout=args.timeout, limits=limits, transport=transport) as client:
            level = await run_level(client, target, clips, concurrency, language=args.language, model=args.model,
                                    headers=headers)
        print(json.dumps({k: v for k, v in level.items() if k != "first_failures"}), flush=True)
        for failure in level["first_failures"]:
            print(f"  failed: {failure}", file=sys.stderr)
        levels.append(level)
    return {"label": args.label, "url": target.url, "provider": target.provider, "clips": len(clips),
            "seed": args.seed, "levels": levels}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--manifest", required=True, help="TSV/CSV/JSONL of audio paths and reference text")
    parser.add_argument("--audio-root", help="directory holding the clips, if not beside the manifest")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="server root (see run_german_eval.py)")
    parser.add_argument("--url", help="full endpoint, e.g. http://localhost:5011/transcribe")
    parser.add_argument("--api", choices=API_CHOICES, default="auto")
    parser.add_argument("--provider", help="STT provider through the gateway's /api/stt, e.g. vllm-asr")
    parser.add_argument("--api-key", default="", help="bearer token for a gateway with API_KEY")
    parser.add_argument("--model", default="whisper-1", help="advisory, for the /v1 route")
    parser.add_argument("--language", default="de")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32],
                        help="requests in flight at once, one measurement per value (default 1 8 32)")
    parser.add_argument("--limit", type=int, help="a random sample of N clips (run_german_eval's selection)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--label", default="run")
    parser.add_argument("--out", help="write the JSON report here")
    return parser


def main(argv: Optional[list[str]] = None, *, transport=None) -> int:
    args = build_parser().parse_args(argv)
    if any(c < 1 for c in args.concurrency):
        raise SystemExit("--concurrency values are 1 or more")
    report = asyncio.run(_run(args, transport=transport))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return 1 if any(level["failed"] for level in report["levels"]) else 0


if __name__ == "__main__":
    sys.exit(main())
