"""Word/character error rates for German ASR, and whether a difference is real.

Pure stdlib, no dependencies, no I/O — so it is testable offline with no audio,
no model and no GPU. The runner that actually calls the API lives next door in
run_german_eval.py.

WHAT THIS IS FOR
----------------
Three defaults in this project are currently unverified for German:

  * ``WHISPER_COMPUTE_TYPE`` — int8_float16 is auto-selected on GPU because it
    roughly halves VRAM, which is the difference between fitting and not fitting
    on a 12 GB card. Its accuracy cost on German has never been measured here.
  * whisper.cpp quantisation (``q5_0`` / ``q5_1``) on the ARM and Vulkan paths.
  * Qwen3-ASR quantisation.

Each is a memory-vs-accuracy trade, and the priority order of this project makes
the memory side attractive. That is only a safe trade if the accuracy cost is
known, and German is an acceptance criterion.

RELATIVE, NOT ABSOLUTE
----------------------
This measures **one configuration against another on the same audio**. It is not
built to reproduce published WER figures, and it should not be compared against
them: normalisation choices, number formatting and punctuation conventions all
shift absolute WER by more than the differences being measured here.

That focus is what makes the design defensible. Number formatting ("3" versus
"drei") inflates absolute WER badly, but it inflates BOTH configurations
identically, so it cancels in the comparison. No number expansion is attempted,
because a half-correct expander would introduce errors that do *not* cancel.

IS THE DIFFERENCE REAL?
-----------------------
On a 50-utterance set, a 0.3-point WER gap is noise. ``paired_bootstrap``
answers the only question that matters when choosing a default: is B actually
worse than A, or did it just get a different draw? Decide with the confidence
interval, never with the point estimate.
"""

from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

# --- normalisation -----------------------------------------------------------

# Punctuation is dropped rather than compared: ASR punctuation is a formatting
# choice, not a recognition result, and every backend here punctuates
# differently. Hyphens and slashes become SPACES rather than being deleted, so
# "E-Mail-Adresse" and "E Mail Adresse" agree instead of scoring three errors —
# German compounds are hyphenated inconsistently even between humans.
_TO_SPACE = re.compile(r"[-–—/]")
_STRIP = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS = re.compile(r"\s+")

# Typographic variants that carry no phonetic information.
_QUOTES = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "«": '"', "»": '"', "‹": "'", "›": "'",
}

UMLAUT_FOLD = {
    "ä": "ae", "ö": "oe", "ü": "ue",
    "Ä": "ae", "Ö": "oe", "Ü": "ue",
    "ß": "ss",
}


def normalize_german(text: str, *, fold_umlauts: bool = False) -> str:
    """Normalise German text for error-rate comparison.

    Lowercasing uses ``str.lower()``, **not** ``str.casefold()``. Casefold maps
    ``ß`` to ``ss``, which would silently apply half of an umlaut fold whether or
    not one was asked for, and would make ``fold_umlauts`` untestable. Keeping
    that decision explicit is the point.

    ``fold_umlauts`` maps ä→ae, ö→oe, ü→ue, ß→ss. Default False, because those
    distinctions are phonemic in German and a model that writes "Grusse" for
    "Grüße" genuinely got it wrong. Turn it on only when comparing a backend that
    cannot emit umlauts at all — otherwise it hides real errors.
    """
    if not text:
        return ""

    # NFC first: "ü" as u+combining-diaeresis and "ü" as a single codepoint must
    # not count as different words. Without this, a backend that emits
    # decomposed forms scores ~100% WER for a purely cosmetic reason.
    text = unicodedata.normalize("NFC", text)

    for src, dst in _QUOTES.items():
        text = text.replace(src, dst)

    text = text.lower()

    if fold_umlauts:
        for src, dst in UMLAUT_FOLD.items():
            text = text.replace(src.lower(), dst)

    text = _TO_SPACE.sub(" ", text)
    text = _STRIP.sub("", text)
    return _WS.sub(" ", text).strip()


def tokenize(text: str, *, fold_umlauts: bool = False) -> list[str]:
    """Normalised whitespace tokens."""
    normalized = normalize_german(text, fold_umlauts=fold_umlauts)
    return normalized.split() if normalized else []


# --- edit distance -----------------------------------------------------------


@dataclass(frozen=True)
class ErrorCounts:
    """Substitutions, deletions, insertions against a reference length."""

    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    reference_length: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        """Errors per reference token.

        An empty reference with a non-empty hypothesis is 1.0, not infinity or a
        divide-by-zero: every hypothesis token is an insertion, and reporting a
        finite 100% keeps a corpus total from being poisoned by one bad row.
        Empty against empty is 0.0.
        """
        if self.reference_length == 0:
            return 1.0 if self.insertions else 0.0
        return self.errors / self.reference_length

    def __add__(self, other: "ErrorCounts") -> "ErrorCounts":
        return ErrorCounts(
            self.substitutions + other.substitutions,
            self.deletions + other.deletions,
            self.insertions + other.insertions,
            self.reference_length + other.reference_length,
        )


def _levenshtein(reference: Sequence, hypothesis: Sequence) -> ErrorCounts:
    """Edit distance with an operation breakdown.

    Two rows rather than a full matrix: a long-form reference can run to
    thousands of tokens, and the full table is not needed since only the counts
    are reported, not the alignment path. Each cell carries its own counts so
    the breakdown follows whichever path the distance actually took.
    """
    n, m = len(reference), len(hypothesis)
    if n == 0:
        return ErrorCounts(insertions=m, reference_length=0)
    if m == 0:
        return ErrorCounts(deletions=n, reference_length=n)

    # row[j] = (distance, ErrorCounts) for reference[:i] vs hypothesis[:j]
    previous: list[tuple[int, ErrorCounts]] = [
        (j, ErrorCounts(insertions=j)) for j in range(m + 1)
    ]

    for i in range(1, n + 1):
        current: list[tuple[int, ErrorCounts]] = [(i, ErrorCounts(deletions=i))]
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                cost, counts = previous[j - 1]
                current.append((cost, counts))
                continue

            sub_cost, sub_counts = previous[j - 1]
            del_cost, del_counts = previous[j]
            ins_cost, ins_counts = current[j - 1]

            best = min(sub_cost, del_cost, ins_cost)
            if best == sub_cost:
                current.append((best + 1, ErrorCounts(
                    sub_counts.substitutions + 1, sub_counts.deletions, sub_counts.insertions)))
            elif best == del_cost:
                current.append((best + 1, ErrorCounts(
                    del_counts.substitutions, del_counts.deletions + 1, del_counts.insertions)))
            else:
                current.append((best + 1, ErrorCounts(
                    ins_counts.substitutions, ins_counts.deletions, ins_counts.insertions + 1)))
        previous = current

    _, counts = previous[m]
    return ErrorCounts(
        counts.substitutions, counts.deletions, counts.insertions, reference_length=n
    )


def word_errors(reference: str, hypothesis: str, *, fold_umlauts: bool = False) -> ErrorCounts:
    """Word-level error counts between two strings."""
    return _levenshtein(
        tokenize(reference, fold_umlauts=fold_umlauts),
        tokenize(hypothesis, fold_umlauts=fold_umlauts),
    )


def character_errors(reference: str, hypothesis: str, *, fold_umlauts: bool = False) -> ErrorCounts:
    """Character-level error counts, spaces removed.

    CER is the more informative metric for German. A single wrong morpheme in a
    compound ("Rechtschreibprüfung" vs "Rechtschreibprufung") is one whole word
    error out of few words, so WER swings hard on long compounds; CER degrades
    proportionally to what was actually misheard.
    """
    ref = normalize_german(reference, fold_umlauts=fold_umlauts).replace(" ", "")
    hyp = normalize_german(hypothesis, fold_umlauts=fold_umlauts).replace(" ", "")
    return _levenshtein(ref, hyp)


def corpus_rate(pairs: Iterable[tuple[str, str]], *, fold_umlauts: bool = False,
                character_level: bool = False) -> ErrorCounts:
    """Aggregate error counts over (reference, hypothesis) pairs.

    Totals are summed before dividing — the corpus rate is total errors over
    total reference tokens, NOT the mean of per-utterance rates. Averaging rates
    would weight a three-word utterance the same as a sixty-word one.
    """
    measure = character_errors if character_level else word_errors
    total = ErrorCounts()
    for reference, hypothesis in pairs:
        total = total + measure(reference, hypothesis, fold_umlauts=fold_umlauts)
    return total


# --- is the difference real? -------------------------------------------------


@dataclass
class BootstrapResult:
    """Paired bootstrap comparison of two systems on the same utterances."""

    baseline_rate: float
    candidate_rate: float
    delta: float                      # candidate - baseline; negative = better
    ci_low: float
    ci_high: float
    confidence: float
    samples: int
    resamples: int
    candidate_worse_fraction: float = 0.0
    per_utterance: list[float] = field(default_factory=list)

    @property
    def significant(self) -> bool:
        """True when the interval excludes zero."""
        return self.ci_low > 0.0 or self.ci_high < 0.0

    def verdict(self) -> str:
        """One line a human can act on."""
        direction = "worse" if self.delta > 0 else "better"
        magnitude = f"{abs(self.delta) * 100:.2f} points"
        interval = f"[{self.ci_low * 100:+.2f}, {self.ci_high * 100:+.2f}]"
        if not self.significant:
            return (
                f"NO SIGNIFICANT DIFFERENCE — candidate is {magnitude} {direction}, "
                f"but the {self.confidence:.0%} CI {interval} includes zero "
                f"on {self.samples} utterances. Do not decide on this."
            )
        return (
            f"SIGNIFICANT — candidate is {magnitude} {direction} "
            f"({self.confidence:.0%} CI {interval}, {self.samples} utterances)."
        )


def paired_bootstrap(
    baseline: Sequence[ErrorCounts],
    candidate: Sequence[ErrorCounts],
    *,
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: Optional[int] = 1234,
) -> BootstrapResult:
    """Is the candidate genuinely worse than the baseline, or is it noise?

    PAIRED — each resample draws the same utterance indices from both systems,
    so per-utterance difficulty cancels. An unpaired test on a small set would
    mostly measure which utterances are hard.

    Resampling is over UTTERANCES, and each resample recomputes the corpus rate
    as total-errors-over-total-reference-tokens, matching how the headline number
    is computed. Averaging per-utterance rates instead would answer a different
    question and give a narrower, wrong interval.

    ``seed`` is fixed by default so a reported interval can be reproduced
    exactly; pass None for a fresh draw.
    """
    if len(baseline) != len(candidate):
        raise ValueError(
            f"paired comparison needs equal lengths, got {len(baseline)} and {len(candidate)}"
        )
    if not baseline:
        raise ValueError("nothing to compare: no utterances")

    def rate(counts: Iterable[ErrorCounts]) -> float:
        total = ErrorCounts()
        for c in counts:
            total = total + c
        return total.rate

    baseline_rate = rate(baseline)
    candidate_rate = rate(candidate)
    observed_delta = candidate_rate - baseline_rate

    rng = random.Random(seed)
    n = len(baseline)
    deltas: list[float] = []
    worse = 0
    for _ in range(resamples):
        picks = [rng.randrange(n) for _ in range(n)]
        b = rate(baseline[i] for i in picks)
        c = rate(candidate[i] for i in picks)
        d = c - b
        deltas.append(d)
        if d > 0:
            worse += 1

    deltas.sort()
    tail = (1.0 - confidence) / 2.0
    low = deltas[max(0, int(tail * resamples))]
    high = deltas[min(resamples - 1, int((1.0 - tail) * resamples))]

    return BootstrapResult(
        baseline_rate=baseline_rate,
        candidate_rate=candidate_rate,
        delta=observed_delta,
        ci_low=low,
        ci_high=high,
        confidence=confidence,
        samples=n,
        resamples=resamples,
        candidate_worse_fraction=worse / resamples,
        per_utterance=[c.rate - b.rate for b, c in zip(baseline, candidate)],
    )
