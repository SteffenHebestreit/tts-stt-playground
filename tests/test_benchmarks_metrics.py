"""The faster edit distance and bootstrap must give the answers the slow ones did.

benchmarks/asr_metrics.py used to allocate a dataclass per DP cell, which made a
character comparison of anything long-form intractable. The rewrite (trimmed
common ends, a banded two-row DP over plain ints, an optional rapidfuzz path) is
only acceptable if it is *the same function*: same totals, same
substitution/deletion/insertion split, same tie-breaks, same bootstrap interval.
The oracles below are the previous implementations, kept verbatim in spirit.
"""

from __future__ import annotations

import importlib.util
import random
import sys
import time
import types
from dataclasses import astuple, dataclass
from pathlib import Path

import pytest

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"


@pytest.fixture(scope="module")
def metrics():
    spec = importlib.util.spec_from_file_location(
        "asr_metrics_equivalence_tests", BENCHMARKS / "asr_metrics.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["asr_metrics_equivalence_tests"] = module    # @dataclass looks itself up here
    spec.loader.exec_module(module)
    return module


# --- oracles: the implementations this replaced -----------------------------


@dataclass(frozen=True)
class Counts:
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0


def oracle_levenshtein(reference, hypothesis):
    """Full DP, one counts object per cell; ties go substitution, deletion, insertion."""
    n, m = len(reference), len(hypothesis)
    if n == 0:
        return (0, 0, m)
    if m == 0:
        return (0, n, 0)
    previous = [(j, Counts(insertions=j)) for j in range(m + 1)]
    for i in range(1, n + 1):
        current = [(i, Counts(deletions=i))]
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                current.append(previous[j - 1])
                continue
            sub_cost, sub = previous[j - 1]
            del_cost, dele = previous[j]
            ins_cost, ins = current[j - 1]
            best = min(sub_cost, del_cost, ins_cost)
            if best == sub_cost:
                current.append((best + 1, Counts(sub.substitutions + 1, sub.deletions, sub.insertions)))
            elif best == del_cost:
                current.append((best + 1, Counts(dele.substitutions, dele.deletions + 1, dele.insertions)))
            else:
                current.append((best + 1, Counts(ins.substitutions, ins.deletions, ins.insertions + 1)))
        previous = current
    counts = previous[m][1]
    return (counts.substitutions, counts.deletions, counts.insertions)


def oracle_bootstrap(metrics, baseline, candidate, *, resamples, confidence, seed):
    """The bootstrap as it was: ErrorCounts summed per resample."""
    def rate(counts):
        total = metrics.ErrorCounts()
        for c in counts:
            total = total + c
        return total.rate

    rng = random.Random(seed)
    n = len(baseline)
    deltas, worse = [], 0
    for _ in range(resamples):
        picks = [rng.randrange(n) for _ in range(n)]
        d = rate(candidate[i] for i in picks) - rate(baseline[i] for i in picks)
        deltas.append(d)
        worse += d > 0
    deltas.sort()
    tail = (1.0 - confidence) / 2.0
    return (rate(baseline), rate(candidate), rate(candidate) - rate(baseline),
            deltas[max(0, int(tail * resamples))],
            deltas[min(resamples - 1, int((1.0 - tail) * resamples))],
            worse / resamples)


def split(counts):
    return (counts.substitutions, counts.deletions, counts.insertions)


def mutate(rng, seq, alphabet, edits):
    seq = list(seq)
    for _ in range(edits):
        op = rng.choice("sdi")
        if op == "s" and seq:
            seq[rng.randrange(len(seq))] = rng.choice(alphabet)
        elif op == "d" and seq:
            del seq[rng.randrange(len(seq))]
        else:
            seq.insert(rng.randint(0, len(seq)), rng.choice(alphabet))
    return seq


# --- edit distance equals the old one, operation for operation ---------------


@pytest.mark.parametrize("band_start", [1, 2, 3, 32])
def test_breakdown_matches_the_full_dp_on_random_sequences(metrics, monkeypatch, band_start):
    """Tiny alphabets make ties everywhere, which is where a tie-break would drift.

    band_start=1 forces the second, wider pass on almost every input, so both
    passes are held to the oracle.
    """
    monkeypatch.setattr(metrics, "_BAND_START", band_start)
    rng = random.Random(band_start)
    for _ in range(1500):
        alphabet = rng.choice(["ab", "abc", "abcdefgh"])
        a = [rng.choice(alphabet) for _ in range(rng.randint(0, 16))]
        if rng.random() < 0.5:
            b = mutate(rng, a, alphabet, rng.randint(0, 5))
            if rng.random() < 0.5:                       # a shared head and tail
                head = [rng.choice(alphabet) for _ in range(rng.randint(0, 3))]
                tail = [rng.choice(alphabet) for _ in range(rng.randint(0, 3))]
                a, b = head + a + tail, head + b + tail
        else:
            b = [rng.choice(alphabet) for _ in range(rng.randint(0, 16))]
        for x, y in ((a, b), ("".join(a), "".join(b))):
            got = metrics._levenshtein(x, y)
            assert split(got) == oracle_levenshtein(x, y), (x, y)
            assert got.reference_length == len(x)


def test_breakdown_matches_on_longer_similar_sequences(metrics, monkeypatch):
    """The realistic shape: mostly identical text with a few scattered errors."""
    monkeypatch.setattr(metrics, "_BAND_START", 4)
    rng = random.Random(11)
    alphabet = "abcdefghijklmnopqrstuvwxyzäöüß "
    for edits in (0, 1, 3, 10, 40, 120):
        for _ in range(40):
            a = [rng.choice(alphabet) for _ in range(rng.randint(1, 200))]
            b = mutate(rng, a, alphabet, edits)
            assert split(metrics._levenshtein(a, b)) == oracle_levenshtein(a, b)


def test_word_lists_and_strings_are_both_accepted(metrics):
    words = "der schnelle braune fuchs springt".split()
    other = "der schnelle rote fuchs springt hoch".split()
    assert split(metrics._levenshtein(words, other)) == oracle_levenshtein(words, other)


def test_error_totals_and_lengths_are_consistent(metrics):
    """S+D+I is the distance, and the operations account for both lengths."""
    rng = random.Random(2)
    for _ in range(300):
        a = "".join(rng.choice("abc") for _ in range(rng.randint(0, 30)))
        b = "".join(rng.choice("abc") for _ in range(rng.randint(0, 30)))
        c = metrics._levenshtein(a, b)
        assert c.reference_length == len(a)
        assert len(a) - c.deletions == len(b) - c.insertions      # both equal matches + substitutions


def test_a_second_pass_is_taken_when_the_first_band_was_too_narrow(metrics, monkeypatch):
    """Nothing like each other: the distance far exceeds the starting band."""
    monkeypatch.setattr(metrics, "_BAND_START", 2)
    a, b = "abcdefghij" * 3, "jihgfedcba" * 3 + "xyz"
    assert split(metrics._levenshtein(a, b)) == oracle_levenshtein(a, b)


# --- long-form: the reason for the rewrite -----------------------------------


def test_long_form_character_comparison_is_tractable_and_exact(metrics):
    """12,000 characters with a substitution every 40th: the answer is known exactly.

    The full DP is 144 million cells here, one dataclass each; it did not finish
    in any reasonable time. The '#' never occurs in the reference, so each one
    costs exactly one edit and the alignment has no choice about how.
    """
    rng = random.Random(4)
    reference = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(12000))
    hypothesis = "".join("#" if i % 40 == 20 else ch for i, ch in enumerate(reference))
    started = time.perf_counter()
    counts = metrics._levenshtein(reference, hypothesis)
    elapsed = time.perf_counter() - started
    assert split(counts) == (300, 0, 0)
    assert counts.reference_length == 12000
    assert elapsed < 20, f"took {elapsed:.1f}s; the banded DP should need well under a second or two"


def test_long_identical_input_costs_nothing(metrics):
    text = "wort " * 50000
    started = time.perf_counter()
    assert metrics._levenshtein(text, text).errors == 0
    assert time.perf_counter() - started < 2


def test_character_and_word_errors_still_agree_with_the_oracle_end_to_end(metrics):
    reference = "Die Rechtschreibprüfung meldet keinen Fehler, aber sie übersieht Grüße."
    hypothesis = "Die Rechtschreibprufung meldet einen Fehler aber sie übersieht Grusse"
    assert split(metrics.word_errors(reference, hypothesis)) == oracle_levenshtein(
        metrics.tokenize(reference), metrics.tokenize(hypothesis))
    ref_chars = metrics.normalize_german(reference).replace(" ", "")
    hyp_chars = metrics.normalize_german(hypothesis).replace(" ", "")
    assert split(metrics.character_errors(reference, hypothesis)) == oracle_levenshtein(ref_chars, hyp_chars)


# --- the optional rapidfuzz path ---------------------------------------------


class FakeEditop:
    def __init__(self, tag):
        self.tag = tag


def install_fake_rapidfuzz(monkeypatch, tags, seen=None):
    """A stand-in for rapidfuzz.distance.Levenshtein returning a canned edit script."""
    levenshtein = types.SimpleNamespace(
        editops=lambda a, b: (seen.append((a, b)) if seen is not None else None)
        or [FakeEditop(t) for t in tags])
    distance = types.ModuleType("rapidfuzz.distance")
    distance.Levenshtein = levenshtein
    package = types.ModuleType("rapidfuzz")
    package.distance = distance
    monkeypatch.setitem(sys.modules, "rapidfuzz", package)
    monkeypatch.setitem(sys.modules, "rapidfuzz.distance", distance)


def test_big_inputs_use_rapidfuzz_when_it_is_installed(metrics, monkeypatch):
    seen = []
    install_fake_rapidfuzz(monkeypatch, ["replace", "replace", "delete", "insert", "insert", "insert"], seen)
    monkeypatch.delenv("ASR_METRICS_NO_RAPIDFUZZ", raising=False)
    monkeypatch.setattr(metrics, "_RAPIDFUZZ_MIN_CELLS", 10)
    counts = metrics._levenshtein("abcdefgh", "xyzwvuts")
    assert split(counts) == (2, 1, 3)
    assert counts.reference_length == 8
    assert seen, "rapidfuzz was not consulted"


def test_small_inputs_never_leave_the_pure_python_path(metrics, monkeypatch):
    """So a saved report is the same with and without rapidfuzz installed."""
    seen = []
    install_fake_rapidfuzz(monkeypatch, ["replace"], seen)
    monkeypatch.delenv("ASR_METRICS_NO_RAPIDFUZZ", raising=False)
    counts = metrics._levenshtein("kleine", "kleene")
    assert seen == []
    assert split(counts) == (1, 0, 0)


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_rapidfuzz_can_be_switched_off_for_a_deterministic_split(metrics, monkeypatch, value):
    seen = []
    install_fake_rapidfuzz(monkeypatch, ["replace"], seen)
    monkeypatch.setenv("ASR_METRICS_NO_RAPIDFUZZ", value)
    monkeypatch.setattr(metrics, "_RAPIDFUZZ_MIN_CELLS", 1)
    counts = metrics._levenshtein("abcdefgh", "abcdxfgh")
    assert seen == []
    assert split(counts) == (1, 0, 0)


def test_a_missing_rapidfuzz_falls_back_to_pure_python(metrics, monkeypatch):
    monkeypatch.setitem(sys.modules, "rapidfuzz", None)          # makes `import rapidfuzz...` raise ImportError
    monkeypatch.delenv("ASR_METRICS_NO_RAPIDFUZZ", raising=False)
    monkeypatch.setattr(metrics, "_RAPIDFUZZ_MIN_CELLS", 1)
    assert split(metrics._levenshtein("abcdefgh", "abcdxfgh")) == (1, 0, 0)


def test_real_rapidfuzz_agrees_on_the_total(metrics, monkeypatch):
    """Only the total is promised across backends; a tied alignment may split differently."""
    pytest.importorskip("rapidfuzz")
    monkeypatch.delenv("ASR_METRICS_NO_RAPIDFUZZ", raising=False)
    rng = random.Random(9)
    for _ in range(300):
        alphabet = rng.choice(["ab", "abcd", "abcdefghij"])
        a = [rng.choice(alphabet) for _ in range(rng.randint(0, 40))]
        b = mutate(rng, a, alphabet, rng.randint(0, 15))
        for x, y in ((a, b), ("".join(a), "".join(b))):
            monkeypatch.setattr(metrics, "_RAPIDFUZZ_MIN_CELLS", 0)
            fast = metrics._levenshtein(x, y)
            monkeypatch.setattr(metrics, "_RAPIDFUZZ_MIN_CELLS", 10 ** 12)
            exact = metrics._levenshtein(x, y)
            assert fast.errors == exact.errors, (x, y)
            assert len(x) - fast.deletions == len(y) - fast.insertions


# --- the bootstrap is the same computation, faster ---------------------------


def make_counts(metrics, rng, n, zero_lengths=False):
    rows = []
    for _ in range(n):
        length = 0 if zero_lengths and rng.random() < 0.2 else rng.choice([4, 7, 12, 20])
        rows.append((rng.randint(0, 3), rng.randint(0, 2), rng.randint(0, 2), length))
    return rows


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_bootstrap_interval_is_identical_to_the_old_implementation(metrics, seed):
    rng = random.Random(seed)
    rows = make_counts(metrics, rng, rng.randint(5, 80), zero_lengths=seed % 2 == 0)
    candidate_rows = [(s + rng.randint(0, 2), d, i + rng.randint(0, 1), ln) for s, d, i, ln in rows]
    baseline = [metrics.ErrorCounts(*r) for r in rows]
    candidate = [metrics.ErrorCounts(*r) for r in candidate_rows]

    got = metrics.paired_bootstrap(baseline, candidate, resamples=400, confidence=0.9, seed=seed)
    expected = oracle_bootstrap(metrics, baseline, candidate, resamples=400, confidence=0.9, seed=seed)
    assert (got.baseline_rate, got.candidate_rate, got.delta, got.ci_low, got.ci_high,
            got.candidate_worse_fraction) == expected


def test_bootstrap_with_no_resamples_is_a_clear_error(metrics):
    counts = [metrics.ErrorCounts(1, 0, 0, 5)]
    with pytest.raises(ValueError, match="resamples"):
        metrics.paired_bootstrap(counts, counts, resamples=0)


def test_bootstrap_over_thousands_of_utterances_finishes_promptly(metrics):
    rng = random.Random(0)
    baseline = [metrics.ErrorCounts(rng.randint(0, 2), 0, 0, 12) for _ in range(3000)]
    candidate = [metrics.ErrorCounts(rng.randint(0, 3), 0, 0, 12) for _ in range(3000)]
    started = time.perf_counter()
    result = metrics.paired_bootstrap(baseline, candidate, resamples=300)
    assert time.perf_counter() - started < 15
    assert result.significant


def test_an_exact_tie_is_not_described_as_better(metrics):
    counts = [metrics.ErrorCounts(1, 0, 0, 10) for _ in range(20)]
    verdict = metrics.paired_bootstrap(counts, list(counts), resamples=100).verdict()
    assert "exactly the same" in verdict
    assert "better" not in verdict and "worse" not in verdict
    assert "Do not decide on this" in verdict
