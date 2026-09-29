#!/usr/bin/env python3
"""Measure German ASR accuracy through the /v1 API, and compare configurations.

The stack is used mostly as an API, so this measures the API — not the Python
internals. Whatever the endpoint actually returns to a client is what gets scored,
including any normalisation the gateway applies on the way through.

TYPICAL USE — deciding whether int8 costs German accuracy
---------------------------------------------------------
    # baseline
    WHISPER_COMPUTE_TYPE=float16 docker compose up -d stt-service
    python benchmarks/run_german_eval.py --manifest data/de.tsv --label float16 \
        --limit 500 --out results-float16.json

    # candidate
    WHISPER_COMPUTE_TYPE=int8_float16 docker compose up -d stt-service
    python benchmarks/run_german_eval.py --manifest data/de.tsv --label int8 \
        --limit 500 --out results-int8.json

    python benchmarks/run_german_eval.py --compare results-float16.json results-int8.json

The comparison prints a paired bootstrap interval. Decide with that, not with
the two WER numbers — on a small set the point estimates differ by more than the
effect being measured.

CHOOSING WHAT IS MEASURED
-------------------------
    # the gateway's default STT provider (POST /v1/audio/transcriptions)
    --base-url http://localhost:3000

    # one named provider through the gateway (POST /api/stt): whisper, qwen3-asr,
    # parakeet, canary, whisper-cpp
    --base-url http://localhost:3000 --provider parakeet

    # a backend directly, no gateway in between (POST /transcribe, stt-form-v1)
    --url http://localhost:5005/transcribe

/v1/audio/transcriptions cannot select a provider: it always uses the configured
default and ignores unknown fields, so a "provider" field sent there would be
silently dropped and the run mislabelled. Use --provider (gateway /api/stt) or
--url (the backend) to compare providers.

GETTING AUDIO
-------------
Any (audio, transcript) pairs work. Common Voice German is the usual choice and
its validated TSV is read natively:
    https://commonvoice.mozilla.org/de/datasets
A few hundred utterances is enough to separate configurations; a few thousand is
better. See benchmarks/README.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))

# This tool prints German. A Windows console defaults to a legacy codepage where
# umlauts and the em-dash in the verdict either mojibake or raise
# UnicodeEncodeError mid-run, losing the results. Force UTF-8 on both streams.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):  # already wrapped, or not a real tty
        pass

from asr_metrics import (  # noqa: E402
    BootstrapResult,
    ErrorCounts,
    character_errors,
    corpus_rate,
    paired_bootstrap,
    word_errors,
)

DEFAULT_BASE_URL = os.getenv("EVAL_BASE_URL", "http://localhost:3000")
DEFAULT_SEED = 1234
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus", ".webm"}

# The three wire protocols a stack in this project speaks. They differ in the
# multipart field that carries the audio, which is why "just POST it" is not
# enough to point the benchmark at a backend.
API_OPENAI = "openai"        # POST /v1/audio/transcriptions: file, model, language
API_STT_FORM = "stt-form"    # POST /transcribe (stt-form-v1): audio, language
API_GATEWAY = "gateway"      # POST /api/stt on the gateway: audio, provider, language
API_CHOICES = ("auto", API_OPENAI, API_STT_FORM, API_GATEWAY)
_API_PATHS = {
    API_OPENAI: "/v1/audio/transcriptions",
    API_STT_FORM: "/transcribe",
    API_GATEWAY: "/api/stt",
}

# Timing goes through this indirection so a test can drive the clock; nothing
# else reads it. perf_counter, not monotonic: it is the higher-resolution clock
# and latencies here are tens of milliseconds.
_clock = time.perf_counter


# --- manifests ---------------------------------------------------------------


def load_manifest(path: Path, audio_root: Optional[Path] = None) -> list[tuple[Path, str]]:
    """Read (audio_path, reference) pairs from TSV, CSV or JSONL.

    Common Voice ships a `path`/`sentence` TSV whose `path` is a bare filename
    relative to a sibling clips/ directory, so a resolved path is tried in
    several places rather than assumed. Formats are detected by content, since
    Common Voice uses .tsv but other corpora use .csv with the same columns.
    """
    if not path.exists():
        raise SystemExit(f"manifest not found: {path}")

    rows: list[tuple[Path, str]] = []
    root = audio_root or path.parent

    # utf-8-sig: a spreadsheet export starts with a byte-order mark, which would
    # otherwise glue itself onto the first header name ("﻿path") and make
    # the audio column unfindable.
    if path.suffix.lower() == ".jsonl":
        for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON — {exc}")
            if not isinstance(record, dict):
                raise SystemExit(f"{path}:{line_no}: expected a JSON object per line")
            audio = record.get("audio") or record.get("path") or record.get("file")
            text = record.get("text") or record.get("sentence") or record.get("reference")
            if not audio:
                raise SystemExit(
                    f"{path}:{line_no}: need an audio field (audio|path|file) and a "
                    f"text field (text|sentence|reference)"
                )
            if not str(text or "").strip():
                # Same rule as the TSV branch: a row with no reference cannot be
                # scored, and aborting the whole run over it helps nobody.
                continue
            rows.append((_resolve_audio(str(audio), root), str(text)))
        return rows

    with path.open(encoding="utf-8-sig", newline="") as handle:
        # Sniff the header rather than trusting the extension: Common Voice ships
        # tab-separated data in .tsv, but re-exports of the same columns are
        # routinely .csv, and a wrong delimiter yields one giant column whose
        # failure mode is a confusing "no audio column" error.
        first_line = handle.readline()
        handle.seek(0)
        delimiter = "\t" if "\t" in first_line else ","
        # Tab-separated corpora are not quoted: Common Voice sentences contain
        # bare double quotes, and with the default quoting a sentence that opens
        # with one swallows the following tabs and rows into a single field.
        quoting = csv.QUOTE_NONE if delimiter == "\t" else csv.QUOTE_MINIMAL
        reader = csv.DictReader(handle, delimiter=delimiter, quoting=quoting)
        if not reader.fieldnames:
            raise SystemExit(f"{path}: no header row")

        audio_col = _pick_column(reader.fieldnames, ("path", "audio", "file", "filename"))
        text_col = _pick_column(reader.fieldnames, ("sentence", "text", "reference", "transcript"))
        if not audio_col or not text_col:
            raise SystemExit(
                f"{path}: could not find an audio column and a text column in "
                f"{reader.fieldnames}. Expected one of path/audio/file/filename and "
                f"one of sentence/text/reference/transcript."
            )
        for row in reader:
            audio = (row.get(audio_col) or "").strip()
            text = (row.get(text_col) or "").strip()
            if not audio or not text:
                continue  # Common Voice has rows with no validated sentence
            rows.append((_resolve_audio(audio, root), text))
    return rows


def _pick_column(fieldnames: list[str], candidates: tuple[str, ...]) -> Optional[str]:
    lowered = {name.lower().strip(): name for name in fieldnames}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _resolve_audio(value: str, root: Path) -> Path:
    """Find the clip, trying the layouts real corpora actually use."""
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    for base in (root, root / "clips", root.parent, root.parent / "clips"):
        resolved = base / candidate
        if resolved.exists():
            return resolved
    return root / candidate  # report the miss later, with a useful path


def sample_manifest(
    rows: list[tuple[Path, str]], limit: Optional[int], seed: int
) -> list[tuple[Path, str]]:
    """A reproducible random subset of `limit` rows, kept in manifest order.

    Corpora are not shuffled: Common Voice's validated.tsv is grouped by client,
    so "the first N" is a handful of speakers, and a quantisation effect that
    only shows on some voices can vanish or be exaggerated depending on whose
    clips come first. The same seed over the same manifest always yields the same
    clips, so a baseline and a candidate run see identical audio — that is what
    makes the comparison paired.
    """
    if not limit or limit >= len(rows):
        return list(rows)
    chosen = sorted(random.Random(seed).sample(range(len(rows)), limit))
    return [rows[index] for index in chosen]


# --- what to talk to ---------------------------------------------------------


@dataclass(frozen=True)
class Target:
    """Where a transcription request goes and which wire protocol it uses."""

    api: str
    url: str                          # the full endpoint
    base_url: str                     # scheme://host:port, for the report
    provider: Optional[str] = None


def _infer_api(path: str) -> Optional[str]:
    tail = path.rstrip("/")
    # whisper.cpp's own server calls the same thing /inference.
    if tail.endswith("/audio/transcriptions") or tail.endswith("/inference"):
        return API_OPENAI
    if tail.endswith("/transcribe"):
        return API_STT_FORM
    if tail.endswith("/api/stt"):
        return API_GATEWAY
    return None


def resolve_target(
    base_url: str,
    url: Optional[str] = None,
    api: str = "auto",
    provider: Optional[str] = None,
) -> Target:
    """Work out the endpoint and protocol from --base-url / --url / --api / --provider."""
    provider = (provider or "").strip() or None
    full_url: Optional[str] = None

    if url:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise SystemExit(
                f"--url must be an absolute http(s) URL such as "
                f"http://localhost:5001/transcribe, got {url!r}"
            )
        root = f"{parts.scheme}://{parts.netloc}"
        # A bare host is a base URL, not an endpoint: fall through to the
        # per-protocol path below.
        if parts.path.strip("/"):
            full_url = url
    else:
        root = base_url.rstrip("/")

    resolved = api
    if api == "auto":
        resolved = _infer_api(urlsplit(full_url).path) if full_url else None
        if resolved is None:
            if full_url:
                raise SystemExit(
                    f"cannot tell which API {full_url} speaks; pass --api "
                    f"{'|'.join(API_CHOICES[1:])}"
                )
            resolved = API_GATEWAY if provider else API_OPENAI

    if provider and resolved != API_GATEWAY:
        raise SystemExit(
            f"--provider cannot be used with the '{resolved}' API. "
            f"POST /v1/audio/transcriptions always uses the gateway's default provider "
            f"and drops unknown fields, so the run would be labelled with a provider it "
            f"never measured. Use the gateway's /api/stt (drop --url or pass "
            f"--api gateway) to pick a provider, or --url http://<backend>/transcribe "
            f"to talk to one directly."
        )
    if resolved == API_GATEWAY and not provider:
        raise SystemExit("the gateway API needs --provider (whisper, qwen3-asr, parakeet, canary, whisper-cpp)")

    endpoint = full_url or root + _API_PATHS[resolved]
    return Target(api=resolved, url=endpoint, base_url=root, provider=provider)


# --- transcription -----------------------------------------------------------


class FallbackResponse(RuntimeError):
    """The gateway answered from a provider other than the one being measured."""


@dataclass
class Transcription:
    text: str
    seconds: float
    audio_seconds: Optional[float] = None
    served_by: Optional[str] = None


def _request_fields(target: Target, *, model: str, language: str) -> tuple[str, dict]:
    """(multipart field carrying the audio, extra form data) for this protocol."""
    if target.api == API_OPENAI:
        return "file", {"model": model, "language": language, "response_format": "json"}
    if target.api == API_GATEWAY:
        return "audio", {"provider": target.provider, "language": language}
    return "audio", {"language": language}


def _extract_text(payload: dict) -> str:
    text = payload.get("text") or payload.get("transcript") or ""
    segments = payload.get("segments")
    if not text and isinstance(segments, list):
        text = " ".join(str(s.get("text", "")) for s in segments if isinstance(s, dict))
    return str(text).strip()


def _audio_seconds(payload: dict) -> Optional[float]:
    """The clip length, when the backend states it (the stt-form backends do)."""
    try:
        value = float(payload.get("duration"))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def transcribe(client, target: Target, audio: Path, *, model: str, language: str,
               timeout: float, api_key: str = "") -> Transcription:
    """POST one clip to the target endpoint."""
    field, data = _request_fields(target, model=model, language=language)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    started = _clock()
    with audio.open("rb") as handle:
        response = client.post(
            target.url,
            files={field: (audio.name, handle, "application/octet-stream")},
            data=data,
            headers=headers,
            timeout=timeout,
        )
    elapsed = _clock() - started
    if response.status_code != 200:
        raise RuntimeError(f"{audio.name}: HTTP {response.status_code} — {response.text[:200]}")

    # The gateway falls back to another provider when the configured default is
    # down, and says so only in a header. Scoring that answer would attribute
    # another model's errors to the one this run is labelled with.
    fallback = response.headers.get("X-Provider-Fallback")
    served_by = response.headers.get("X-Provider")
    if not fallback and target.provider and served_by and served_by != target.provider:
        fallback = f"{target.provider}->{served_by}"
    if fallback:
        raise FallbackResponse(
            f"{audio.name}: answered by a fallback provider ({fallback}), not the one being "
            f"measured — not scored"
        )
    try:
        payload = response.json()
    except ValueError:
        raise RuntimeError(f"{audio.name}: response was not JSON: {response.text[:120]!r}")
    if not isinstance(payload, dict):
        raise RuntimeError(f"{audio.name}: unexpected response shape {type(payload).__name__}")

    return Transcription(
        text=_extract_text(payload),
        seconds=elapsed,
        audio_seconds=_audio_seconds(payload),
        served_by=served_by,
    )


def _latency_summary(seconds: list[float]) -> dict:
    if not seconds:
        return {}
    ordered = sorted(seconds)
    # Nearest-rank p95: with a few hundred requests interpolation buys nothing.
    p95 = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": p95,
        "max": ordered[-1],
    }


def run(args: argparse.Namespace, *, client=None) -> int:
    """Run one configuration. `client` is any object with an httpx-style `post`."""
    owned_client = client is None
    if owned_client:
        try:
            import httpx
        except ImportError:
            raise SystemExit("httpx is required to run an evaluation: pip install httpx")
        client = httpx.Client()
    try:
        return _run(args, client)
    finally:
        if owned_client:
            client.close()


def _run(args: argparse.Namespace, client) -> int:
    target = resolve_target(args.base_url, args.url, args.api, args.provider)

    all_rows = load_manifest(Path(args.manifest), Path(args.audio_root) if args.audio_root else None)
    manifest = sample_manifest(all_rows, args.limit, args.seed)
    if not manifest:
        raise SystemExit("manifest produced no usable rows")

    missing = [p for p, _ in manifest if not p.exists()]
    if missing:
        preview = "\n  ".join(str(p) for p in missing[:5])
        raise SystemExit(
            f"{len(missing)} audio file(s) not found, e.g.:\n  {preview}\n"
            f"Pass --audio-root to point at the directory holding the clips."
        )

    provider_note = f", provider={target.provider}" if target.provider else ""
    sample_note = (f" (random sample of {len(all_rows)}, seed {args.seed})"
                   if len(manifest) < len(all_rows) else "")
    print(f"{len(manifest)} utterances{sample_note} -> {target.url} "
          f"(api={target.api}{provider_note}, model={args.model}, language={args.language})",
          file=sys.stderr)

    call = dict(model=args.model, language=args.language, timeout=args.timeout,
                api_key=args.api_key)
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Warm up on the first clip, untimed and unscored. Idle-unload TTLs mean the
    # first request after a pause pays a full model load (seconds, versus tens of
    # milliseconds warm), and folding that into the totals made a run's latency
    # depend on how long the service had been idle. The cold figure is reported
    # separately instead of being thrown away, since it is the number that
    # matters for a service that unloads.
    warmup_seconds: list[float] = []
    if args.warmup > 0:
        warm_clip = manifest[0][0]
        for attempt in range(1, args.warmup + 1):
            try:
                warm = transcribe(client, target, warm_clip, **call)
            except Exception as exc:  # noqa: BLE001
                print(f"  warmup {attempt}/{args.warmup}: FAILED — {exc}", file=sys.stderr)
                if args.strict:
                    return 1
                break
            warmup_seconds.append(warm.seconds)
        if warmup_seconds:
            print(f"  warmup: first (cold) request {warmup_seconds[0]:.2f} s"
                  + (f", then {statistics.median(warmup_seconds[1:]):.2f} s"
                     if len(warmup_seconds) > 1 else ""), file=sys.stderr)
    else:
        print("  no warmup: the first request's latency includes any model load", file=sys.stderr)

    results = []
    errors: list[dict] = []
    fallback_responses = 0
    total_wall_seconds = 0.0
    interrupted = False
    for index, (audio, reference) in enumerate(manifest, 1):
        try:
            got = transcribe(client, target, audio, **call)
        except KeyboardInterrupt:
            # Hours of results are worth more than a traceback: score what ran.
            interrupted = True
            print(f"  interrupted after {len(results)} of {len(manifest)} utterances; "
                  f"reporting those", file=sys.stderr)
            break
        except Exception as exc:  # noqa: BLE001 — one bad clip must not lose the run
            if isinstance(exc, FallbackResponse):
                fallback_responses += 1
            print(f"  [{index}/{len(manifest)}] {audio.name}: FAILED — {exc}", file=sys.stderr)
            errors.append({"audio": str(audio), "error": str(exc)})
            if args.strict:
                return 1
            continue

        total_wall_seconds += got.seconds
        wer = word_errors(reference, got.text, fold_umlauts=args.fold_umlauts)
        cer = character_errors(reference, got.text, fold_umlauts=args.fold_umlauts)
        record = {
            "audio": str(audio),
            "reference": reference,
            "hypothesis": got.text,
            "wer": asdict(wer),
            "cer": asdict(cer),
            "seconds": got.seconds,
        }
        if got.audio_seconds is not None:
            record["audio_seconds"] = got.audio_seconds
        results.append(record)
        if index % 25 == 0 or index == len(manifest):
            partial = corpus_rate(
                [(r["reference"], r["hypothesis"]) for r in results],
                fold_umlauts=args.fold_umlauts,
            )
            print(f"  [{index}/{len(manifest)}] running WER {partial.rate:.2%}", file=sys.stderr)

    if not results:
        hint = (" (every response came from a fallback provider: the one you asked for is down)"
                if fallback_responses == len(errors) and fallback_responses else "")
        raise SystemExit(f"every utterance failed; nothing to report{hint}")

    pairs = [(r["reference"], r["hypothesis"]) for r in results]
    wer_total = corpus_rate(pairs, fold_umlauts=args.fold_umlauts)
    cer_total = corpus_rate(pairs, fold_umlauts=args.fold_umlauts, character_level=True)

    latency = _latency_summary([r["seconds"] for r in results])
    timed = [r for r in results if "audio_seconds" in r]
    audio_total = sum(r["audio_seconds"] for r in timed)
    # Only where the backend said how long the clip is: the stt-form backends
    # return `duration`, the gateway's /v1 route does not. Guessing it from the
    # file size would be wrong for every compressed format.
    rtf = (sum(r["seconds"] for r in timed) / audio_total) if timed and audio_total else None

    report = {
        "label": args.label,
        "base_url": target.base_url,
        "url": target.url,
        "api": target.api,
        "provider": target.provider,
        "model": args.model,
        "language": args.language,
        "fold_umlauts": args.fold_umlauts,
        "manifest": str(args.manifest),
        "manifest_rows": len(all_rows),
        "limit": args.limit,
        "seed": args.seed,
        "started_at": started_at,
        "interrupted": interrupted,
        "utterances": len(results),
        "failed": len(errors),
        "not_run": len(manifest) - len(results) - len(errors),
        "fallback_responses": fallback_responses,
        "errors": errors,
        "wer": wer_total.rate,
        "cer": cer_total.rate,
        "wer_counts": asdict(wer_total),
        "cer_counts": asdict(cer_total),
        # Sum of the timed per-request latencies, warmup excluded. Not audio
        # duration; see rtf for that.
        "wall_seconds": total_wall_seconds,
        "latency": latency,
        "warmup": {"requests": args.warmup, "seconds": warmup_seconds},
        # First warmup request: what a caller pays when the model was unloaded.
        "cold_start_seconds": warmup_seconds[0] if warmup_seconds else None,
        "audio_seconds": audio_total if timed else None,
        "audio_seconds_clips": len(timed),
        "rtf": rtf,
        "results": results,
    }

    print(f"\n{args.label}: WER {wer_total.rate:.2%}  CER {cer_total.rate:.2%}  "
          f"({len(results)} utterances, {report['failed']} failed)")
    print(f"  substitutions={wer_total.substitutions} deletions={wer_total.deletions} "
          f"insertions={wer_total.insertions}")
    print(f"  latency: warm median {latency['median']:.2f} s, p95 {latency['p95']:.2f} s"
          + (f"; cold start {warmup_seconds[0]:.2f} s" if warmup_seconds else ""))
    if rtf is not None:
        print(f"  RTF {rtf:.3f} (request time / audio duration, {len(timed)} clips with a stated duration)")
    if fallback_responses:
        print(f"  WARNING: {fallback_responses} response(s) came from a fallback provider and were "
              f"not scored; this run does not fully measure '{args.label}'")

    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  written to {args.out}")
    return 0


# --- comparison --------------------------------------------------------------


def _counts_from(report: dict, key: str) -> list[ErrorCounts]:
    return [ErrorCounts(**r[key]) for r in report["results"]]


def _keyed(results: list[dict]) -> dict:
    """Results by (audio, nth occurrence), so a clip listed twice is not collapsed."""
    seen: dict[str, int] = {}
    keyed = {}
    for record in results:
        nth = seen.get(record["audio"], 0)
        seen[record["audio"]] = nth + 1
        keyed[(record["audio"], nth)] = record
    return keyed


def compare(baseline_path: str, candidate_path: str, *, resamples: int,
            character_level: bool, show: int) -> int:
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    candidate = json.loads(Path(candidate_path).read_text(encoding="utf-8"))

    # Per-utterance counts are baked in at run time, so two runs scored under
    # different normalisation are not comparable no matter how they are paired.
    if bool(baseline.get("fold_umlauts")) != bool(candidate.get("fold_umlauts")):
        raise SystemExit(
            "the two runs were scored with different --fold-umlauts settings; their "
            "error counts are not comparable. Re-run one of them."
        )

    # Align on the audio path. Two runs over the same manifest can still differ
    # in length if a clip failed in one of them, and silently comparing unequal
    # lists would pair unrelated utterances.
    by_audio = _keyed(candidate["results"])
    paired = [(b, by_audio[key]) for key, b in _keyed(baseline["results"]).items() if key in by_audio]
    dropped = len(baseline["results"]) - len(paired)
    extra = len(candidate["results"]) - len(paired)
    if not paired:
        raise SystemExit("the two runs share no utterances — were they run on the same manifest?")
    if dropped:
        print(f"note: {dropped} utterance(s) present in the baseline but not the candidate; "
              f"comparing the {len(paired)} they share", file=sys.stderr)
    if extra:
        print(f"note: {extra} utterance(s) present in the candidate but not the baseline; "
              f"comparing the {len(paired)} they share", file=sys.stderr)
    changed_reference = sum(1 for b, c in paired if b["reference"] != c["reference"])
    if changed_reference:
        print(f"note: {changed_reference} shared clip(s) have different reference text in the two "
              f"runs — different manifests?", file=sys.stderr)
    for name, report in (("baseline", baseline), ("candidate", candidate)):
        if report.get("fallback_responses"):
            print(f"note: {report['fallback_responses']} {name} response(s) came from a fallback "
                  f"provider and were left out of that run", file=sys.stderr)
        if report.get("interrupted"):
            print(f"note: the {name} run was interrupted; it is a partial result", file=sys.stderr)

    key = "cer" if character_level else "wer"
    result = paired_bootstrap(
        [ErrorCounts(**b[key]) for b, _ in paired],
        [ErrorCounts(**c[key]) for _, c in paired],
        resamples=resamples,
    )

    metric = key.upper()
    print(f"baseline  {baseline.get('label', baseline_path):<20} {metric} {result.baseline_rate:.2%}")
    print(f"candidate {candidate.get('label', candidate_path):<20} {metric} {result.candidate_rate:.2%}")
    print()
    print(result.verdict())
    print(f"  candidate was worse in {result.candidate_worse_fraction:.0%} of {resamples} resamples")

    base_latency = (baseline.get("latency") or {}).get("median")
    cand_latency = (candidate.get("latency") or {}).get("median")
    if base_latency is not None and cand_latency is not None:
        print(f"  warm median latency: {base_latency:.2f} s -> {cand_latency:.2f} s")
    base_cold, cand_cold = baseline.get("cold_start_seconds"), candidate.get("cold_start_seconds")
    if base_cold is not None and cand_cold is not None:
        print(f"  cold start:          {base_cold:.2f} s -> {cand_cold:.2f} s")

    if show:
        # Only utterances that really got worse: a header promising "the worst 5"
        # above an empty list, or above unchanged utterances, would mislead.
        changes = [
            (c[key]["substitutions"] + c[key]["deletions"] + c[key]["insertions"]
             - b[key]["substitutions"] - b[key]["deletions"] - b[key]["insertions"], b, c)
            for b, c in paired
        ]
        regressions = sorted((t for t in changes if t[0] > 0), key=lambda t: -t[0])[:show]
        if regressions:
            print(f"\nWorst {len(regressions)} regressions:")
        for delta, b, c in regressions:
            print(f"  +{delta} errors  {Path(b['audio']).name}")
            print(f"      ref: {b['reference']}")
            print(f"      base: {b['hypothesis']}")
            print(f"      cand: {c['hypothesis']}")
    return 0


# --- command line ------------------------------------------------------------


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return number


def _non_negative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError(f"must be 0 or more, got {value}")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure German ASR accuracy through the /v1 API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--manifest", help="TSV/CSV/JSONL of audio paths and reference text")
    parser.add_argument("--audio-root", help="directory holding the clips, if not beside the manifest")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help=f"server root; the endpoint path is added for the chosen API "
                             f"(default {DEFAULT_BASE_URL}, or $EVAL_BASE_URL)")
    parser.add_argument("--url", help="full endpoint to POST to, e.g. http://localhost:5005/transcribe "
                                      "for a backend directly, or .../v1/audio/transcriptions; wins over "
                                      "--base-url")
    parser.add_argument("--api", choices=API_CHOICES, default="auto",
                        help="wire protocol: openai (file+model), stt-form (audio, the backends' "
                             "/transcribe), gateway (audio+provider, the gateway's /api/stt). "
                             "auto picks from --url, and gateway when --provider is given.")
    parser.add_argument("--provider", help="STT provider to measure through the gateway "
                                           "(whisper, qwen3-asr, parakeet, canary, whisper-cpp)")
    parser.add_argument("--api-key", default=os.getenv("EVAL_API_KEY", ""),
                        help="bearer token, for a gateway started with API_KEY (default $EVAL_API_KEY)")
    parser.add_argument("--model", default="whisper-1", help="advisory; the deployment serves what it has")
    parser.add_argument("--language", default="de")
    parser.add_argument("--label", default="run", help="name for this configuration in reports")
    parser.add_argument("--out", help="write the full JSON report here")
    parser.add_argument("--limit", type=_positive_int,
                        help="a random sample of N utterances (see --seed), not the first N")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"seed for the --limit sample; use the same one for every run you "
                             f"intend to compare (default {DEFAULT_SEED})")
    parser.add_argument("--warmup", type=_non_negative_int, default=1, metavar="N",
                        help="untimed, unscored requests before the run, so a model that was "
                             "unloaded is not counted in the latency; the first is reported as "
                             "the cold start (default 1, 0 disables)")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--strict", action="store_true", help="abort on the first failed clip")
    parser.add_argument(
        "--fold-umlauts", action="store_true",
        help="map ä→ae, ö→oe, ü→ue, ß→ss before scoring. Hides real errors; only for a "
             "backend that cannot emit umlauts at all.",
    )
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE", "CANDIDATE"),
                        help="compare two saved reports with a paired bootstrap")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--character-level", action="store_true", help="compare CER instead of WER")
    parser.add_argument("--show-regressions", type=int, default=5, metavar="N")
    return parser


def main(argv: Optional[list[str]] = None, *, client=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.compare:
        return compare(*args.compare, resamples=args.resamples,
                       character_level=args.character_level, show=args.show_regressions)
    if not args.manifest:
        parser.error("--manifest is required unless --compare is used")
    return run(args, client=client)


if __name__ == "__main__":
    raise SystemExit(main())
