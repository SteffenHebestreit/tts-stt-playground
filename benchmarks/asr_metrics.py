"""Word/character error rates for German ASR, and whether a difference is real.

Pure stdlib, no I/O — so it is testable offline with no audio, no model and no
GPU. ``rapidfuzz`` is used, if it happens to be installed, only to align very
long inputs (a long-form character comparison); nothing needs it. The runner
that actually calls the API lives next door in run_german_eval.py.

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

import os
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


# Above this many DP cells (after trimming) a pure-Python pass takes seconds per
# utterance, which is where a long-form character comparison lives. rapidfuzz, if
# it happens to be installed, does the same job in C. It is deliberately NOT used
# below the threshold: it returns *an* optimal alignment, and when several tie the
# substitution/deletion/insertion split can differ from the pure-Python one (the
# error TOTAL never does). Keeping every ordinary utterance on one code path means
# a saved report is the same on a machine with and without rapidfuzz.
_RAPIDFUZZ_MIN_CELLS = 4_000_000
_BAND_START = 32


def _rapidfuzz_levenshtein():
    """rapidfuzz's Levenshtein module, or None when unavailable or switched off."""
    if os.getenv("ASR_METRICS_NO_RAPIDFUZZ", "").strip().lower() in {"1", "true", "yes", "on"}:
        return None
    try:
        from rapidfuzz.distance import Levenshtein
    except ImportError:
        return None
    return Levenshtein


def _levenshtein(reference: Sequence, hypothesis: Sequence) -> ErrorCounts:
    """Edit distance with an operation breakdown.

    Exactly the counts a plain full-matrix DP with the tie-break "substitute,
    else delete, else insert" produces, computed much faster:

    * A common prefix and suffix are cut off first. Both are provably neutral:
      D(xu, xv) == D(u, v), and the DP cells along a matching tail just copy their
      diagonal neighbour. Real ASR output is mostly identical to its reference, so
      this alone removes most of a short utterance.
    * The DP then runs inside a diagonal band. A path that strays k cells from the
      diagonal already costs k, so once an alignment of cost c is known, nothing
      outside a band of width c can beat it. A first pass with a narrow band gives
      such a c, and a second pass with band c is exact. The result is identical to
      the unbanded DP, including the tie-break, because the winning path never
      leaves the band. The work is O(n * distance) instead of O(n * m), which is
      what makes a long-form CER tractable.
    * Each cell holds two plain ints (cost, and the three operation counts packed
      into one) instead of a dataclass, which was the dominant cost.
    """
    n, m = len(reference), len(hypothesis)
    if n == 0:
        return ErrorCounts(insertions=m, reference_length=0)
    if m == 0:
        return ErrorCounts(deletions=n, reference_length=n)

    start = 0
    limit = min(n, m)
    while start < limit and reference[start] == hypothesis[start]:
        start += 1
    end_r, end_h = n, m
    while end_r > start and end_h > start and reference[end_r - 1] == hypothesis[end_h - 1]:
        end_r -= 1
        end_h -= 1
    ref, hyp = reference[start:end_r], hypothesis[start:end_h]

    counts = _trimmed_counts(ref, hyp)
    return ErrorCounts(*counts, reference_length=n)


def _trimmed_counts(ref: Sequence, hyp: Sequence) -> tuple[int, int, int]:
    """(substitutions, deletions, insertions) between two sequences with no shared ends."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return 0, 0, m
    if m == 0:
        return 0, n, 0

    if n * m >= _RAPIDFUZZ_MIN_CELLS:
        fast = _rapidfuzz_levenshtein()
        if fast is not None:
            ops = fast.editops(ref, hyp)
            tags = [op.tag for op in ops]
            return tags.count("replace"), tags.count("delete"), tags.count("insert")

    band = max(abs(n - m), _BAND_START)
    cost, packed, width = _banded_dp(ref, hyp, band)
    if cost > band:
        # The first band was too narrow to be sure. `cost` is a real alignment, so
        # the optimum is at most that, and a band of that width contains it.
        cost, packed, width = _banded_dp(ref, hyp, cost)
    mask = (1 << width) - 1
    return packed >> (2 * width), (packed >> width) & mask, packed & mask


def _banded_dp(ref: Sequence, hyp: Sequence, band: int) -> tuple[int, int, int]:
    """Levenshtein DP restricted to |i - j| <= band. Returns (cost, packed counts, width).

    Counts are packed as substitutions << 2w | deletions << w | insertions, w wide
    enough that no field can overflow into its neighbour. Cells outside the band
    are +infinity, so a value read from one is never chosen.
    """
    n, m = len(ref), len(hyp)
    width = max(n, m).bit_length() + 1
    sub_step, del_step = 1 << (2 * width), 1 << width
    infinity = n + m + band + 2

    last = min(m, band)
    prev_cost = [0] * (m + 1)
    prev_ops = [0] * (m + 1)
    cur_cost = [0] * (m + 1)
    cur_ops = [0] * (m + 1)
    for j in range(last + 1):
        prev_cost[j] = j
        prev_ops[j] = j                      # j insertions
    if last < m:
        prev_cost[last + 1] = infinity

    for i in range(1, n + 1):
        ref_i = ref[i - 1]
        lo = i - band if i - band > 1 else 1
        hi = i + band if i + band < m else m
        if i <= band:
            cur_cost[0] = left_cost = i
            cur_ops[0] = left_ops = i * del_step     # i deletions
        else:
            left_cost, left_ops = infinity, 0
        for j in range(lo, hi + 1):
            if ref_i == hyp[j - 1]:
                left_cost = prev_cost[j - 1]
                left_ops = prev_ops[j - 1]
            else:
                diag, up = prev_cost[j - 1], prev_cost[j]
                if diag <= up and diag <= left_cost:
                    left_cost = diag + 1
                    left_ops = prev_ops[j - 1] + sub_step
                elif up <= left_cost:
                    left_cost = up + 1
                    left_ops = prev_ops[j] + del_step
                else:
                    left_cost += 1
                    left_ops += 1
            cur_cost[j] = left_cost
            cur_ops[j] = left_ops
        if hi < m:
            cur_cost[hi + 1] = infinity
        prev_cost, cur_cost = cur_cost, prev_cost
        prev_ops, cur_ops = cur_ops, prev_ops

    return prev_cost[m], prev_ops[m], width


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
            # An exact tie reads as "0.00 points better" otherwise, which is
            # neither true nor false enough to be worth printing.
            change = (f"candidate is {magnitude} {direction}" if self.delta != 0
                      else "candidate scores exactly the same")
            return (
                f"NO SIGNIFICANT DIFFERENCE — {change}, "
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

    if resamples < 1:
        raise ValueError(f"resamples must be at least 1, got {resamples}")

    # Plain int columns instead of ErrorCounts: a resample sums n rows twice, and
    # summing dataclasses (one allocation per addition) made a 2000-resample run
    # over a few thousand utterances take minutes. The arithmetic is the same
    # total-errors-over-total-reference-tokens as ErrorCounts.rate.
    base_errors = [c.errors for c in baseline]
    cand_errors = [c.errors for c in candidate]
    lengths = [b.reference_length for b in baseline]
    cand_lengths = [c.reference_length for c in candidate]
    base_ins = [c.insertions for c in baseline]
    cand_ins = [c.insertions for c in candidate]

    def rate(errors: int, length: int, insertions: int) -> float:
        if length == 0:
            return 1.0 if insertions else 0.0
        return errors / length

    baseline_rate = rate(sum(base_errors), sum(lengths), sum(base_ins))
    candidate_rate = rate(sum(cand_errors), sum(cand_lengths), sum(cand_ins))
    observed_delta = candidate_rate - baseline_rate

    rng = random.Random(seed)
    n = len(baseline)
    deltas: list[float] = []
    worse = 0
    for _ in range(resamples):
        picks = [rng.randrange(n) for _ in range(n)]
        b = rate(sum(map(base_errors.__getitem__, picks)),
                 sum(map(lengths.__getitem__, picks)),
                 sum(map(base_ins.__getitem__, picks)))
        c = rate(sum(map(cand_errors.__getitem__, picks)),
                 sum(map(cand_lengths.__getitem__, picks)),
                 sum(map(cand_ins.__getitem__, picks)))
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
