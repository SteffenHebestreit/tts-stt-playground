#!/usr/bin/env python3
"""Measure German ASR accuracy through the /v1 API, and compare configurations.

The stack is used mostly as an API, so this measures the API — not the Python
internals. Whatever the gateway actually returns to a client is what gets scored,
including any normalisation the gateway applies on the way through.

TYPICAL USE — deciding whether int8 costs German accuracy
---------------------------------------------------------
    # baseline
    WHISPER_COMPUTE_TYPE=float16 docker compose up -d stt-service
    python benchmarks/run_german_eval.py --manifest data/de.tsv --label float16 \
        --out results-float16.json

    # candidate
    WHISPER_COMPUTE_TYPE=int8_float16 docker compose up -d stt-service
    python benchmarks/run_german_eval.py --manifest data/de.tsv --label int8 \
        --out results-int8.json

    python benchmarks/run_german_eval.py --compare results-float16.json results-int8.json

The comparison prints a paired bootstrap interval. Decide with that, not with
the two WER numbers — on a small set the point estimates differ by more than the
effect being measured.

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
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, Optional

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
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus", ".webm"}


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

    if path.suffix.lower() == ".jsonl":
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON — {exc}")
            audio = record.get("audio") or record.get("path") or record.get("file")
            text = record.get("text") or record.get("sentence") or record.get("reference")
            if not audio or text is None:
                raise SystemExit(
                    f"{path}:{line_no}: need an audio field (audio|path|file) and a "
                    f"text field (text|sentence|reference)"
                )
            rows.append((_resolve_audio(str(audio), root), str(text)))
        return rows

    with path.open(encoding="utf-8", newline="") as handle:
        # Sniff the header rather than trusting the extension: Common Voice ships
        # tab-separated data in .tsv, but re-exports of the same columns are
        # routinely .csv, and a wrong delimiter yields one giant column whose
        # failure mode is a confusing "no audio column" error.
        first_line = handle.readline()
        handle.seek(0)
        delimiter = "\t" if "\t" in first_line else ","
        reader = csv.DictReader(handle, delimiter=delimiter)
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


# --- transcription -----------------------------------------------------------


def transcribe(client, base_url: str, audio: Path, *, model: str, language: str,
               timeout: float) -> tuple[str, float]:
    """POST one clip to /v1/audio/transcriptions. Returns (text, seconds)."""
    started = time.monotonic()
    with audio.open("rb") as handle:
        response = client.post(
            f"{base_url.rstrip('/')}/v1/audio/transcriptions",
            files={"file": (audio.name, handle, "application/octet-stream")},
            data={"model": model, "language": language, "response_format": "json"},
            timeout=timeout,
        )
    elapsed = time.monotonic() - started
    if response.status_code != 200:
        raise RuntimeError(f"{audio.name}: HTTP {response.status_code} — {response.text[:200]}")
    return response.json().get("text", ""), elapsed


def run(args: argparse.Namespace) -> int:
    try:
        import httpx
    except ImportError:
        raise SystemExit("httpx is required to run an evaluation: pip install httpx")

    manifest = load_manifest(Path(args.manifest), Path(args.audio_root) if args.audio_root else None)
    if args.limit:
        manifest = manifest[: args.limit]
    if not manifest:
        raise SystemExit("manifest produced no usable rows")

    missing = [p for p, _ in manifest if not p.exists()]
    if missing:
        preview = "\n  ".join(str(p) for p in missing[:5])
        raise SystemExit(
            f"{len(missing)} audio file(s) not found, e.g.:\n  {preview}\n"
            f"Pass --audio-root to point at the directory holding the clips."
        )

    print(f"{len(manifest)} utterances -> {args.base_url} "
          f"(model={args.model}, language={args.language})", file=sys.stderr)

    results = []
    total_wall_seconds = 0.0
    with httpx.Client() as client:
        for index, (audio, reference) in enumerate(manifest, 1):
            try:
                hypothesis, elapsed = transcribe(
                    client, args.base_url, audio,
                    model=args.model, language=args.language, timeout=args.timeout,
                )
            except Exception as exc:  # noqa: BLE001 — one bad clip must not lose the run
                print(f"  [{index}/{len(manifest)}] {audio.name}: FAILED — {exc}", file=sys.stderr)
                if args.strict:
                    return 1
                continue

            total_wall_seconds += elapsed
            wer = word_errors(reference, hypothesis, fold_umlauts=args.fold_umlauts)
            cer = character_errors(reference, hypothesis, fold_umlauts=args.fold_umlauts)
            results.append({
                "audio": str(audio),
                "reference": reference,
                "hypothesis": hypothesis,
                "wer": asdict(wer),
                "cer": asdict(cer),
                "seconds": elapsed,
            })
            if index % 25 == 0 or index == len(manifest):
                partial = corpus_rate(
                    [(r["reference"], r["hypothesis"]) for r in results],
                    fold_umlauts=args.fold_umlauts,
                )
                print(f"  [{index}/{len(manifest)}] running WER {partial.rate:.2%}", file=sys.stderr)

    if not results:
        raise SystemExit("every utterance failed; nothing to report")

    pairs = [(r["reference"], r["hypothesis"]) for r in results]
    wer_total = corpus_rate(pairs, fold_umlauts=args.fold_umlauts)
    cer_total = corpus_rate(pairs, fold_umlauts=args.fold_umlauts, character_level=True)

    report = {
        "label": args.label,
        "base_url": args.base_url,
        "model": args.model,
        "language": args.language,
        "fold_umlauts": args.fold_umlauts,
        "utterances": len(results),
        "failed": len(manifest) - len(results),
        "wer": wer_total.rate,
        "cer": cer_total.rate,
        "wer_counts": asdict(wer_total),
        "cer_counts": asdict(cer_total),
        # Sum of per-request latencies. Not audio duration — this harness never
        # measures that, so it deliberately reports no real-time factor.
        "wall_seconds": total_wall_seconds,
        "results": results,
    }

    print(f"\n{args.label}: WER {wer_total.rate:.2%}  CER {cer_total.rate:.2%}  "
          f"({len(results)} utterances, {report['failed']} failed)")
    print(f"  substitutions={wer_total.substitutions} deletions={wer_total.deletions} "
          f"insertions={wer_total.insertions}")

    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  written to {args.out}")
    return 0


# --- comparison --------------------------------------------------------------


def _counts_from(report: dict, key: str) -> list[ErrorCounts]:
    return [ErrorCounts(**r[key]) for r in report["results"]]


def compare(baseline_path: str, candidate_path: str, *, resamples: int,
            character_level: bool, show: int) -> int:
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    candidate = json.loads(Path(candidate_path).read_text(encoding="utf-8"))

    # Align on the audio path. Two runs over the same manifest can still differ
    # in length if a clip failed in one of them, and silently comparing unequal
    # lists would pair unrelated utterances.
    by_audio = {r["audio"]: r for r in candidate["results"]}
    paired = [(b, by_audio[b["audio"]]) for b in baseline["results"] if b["audio"] in by_audio]
    dropped = len(baseline["results"]) - len(paired)
    if not paired:
        raise SystemExit("the two runs share no utterances — were they run on the same manifest?")
    if dropped:
        print(f"note: {dropped} utterance(s) present in the baseline but not the candidate; "
              f"comparing the {len(paired)} they share", file=sys.stderr)

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

    if show:
        regressions = sorted(
            ((c[key]["substitutions"] + c[key]["deletions"] + c[key]["insertions"]
              - b[key]["substitutions"] - b[key]["deletions"] - b[key]["insertions"], b, c)
             for b, c in paired),
            key=lambda t: -t[0],
        )[:show]
        print(f"\nWorst {len(regressions)} regressions:")
        for delta, b, c in regressions:
            if delta <= 0:
                break
            print(f"  +{delta} errors  {Path(b['audio']).name}")
            print(f"      ref: {b['reference']}")
            print(f"      base: {b['hypothesis']}")
            print(f"      cand: {c['hypothesis']}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure German ASR accuracy through the /v1 API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--manifest", help="TSV/CSV/JSONL of audio paths and reference text")
    parser.add_argument("--audio-root", help="directory holding the clips, if not beside the manifest")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"default {DEFAULT_BASE_URL}")
    parser.add_argument("--model", default="whisper-1", help="advisory; the deployment serves what it has")
    parser.add_argument("--language", default="de")
    parser.add_argument("--label", default="run", help="name for this configuration in reports")
    parser.add_argument("--out", help="write the full JSON report here")
    parser.add_argument("--limit", type=int, help="only the first N utterances")
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

    args = parser.parse_args(argv)

    if args.compare:
        return compare(*args.compare, resamples=args.resamples,
                       character_level=args.character_level, show=args.show_regressions)
    if not args.manifest:
        parser.error("--manifest is required unless --compare is used")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
