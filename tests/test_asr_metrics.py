"""Offline tests for the German ASR error metrics.

No audio, no model, no network. The metrics are what a decision about
WHISPER_COMPUTE_TYPE will rest on, so they are worth pinning down harder than
the thing being measured.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = REPO_ROOT / "benchmarks"


@pytest.fixture(scope="module")
def metrics():
    spec = importlib.util.spec_from_file_location(
        "asr_metrics_under_test", BENCHMARKS / "asr_metrics.py"
    )
    module = importlib.util.module_from_spec(spec)
    # Register before exec: @dataclass resolves string annotations (this module
    # uses `from __future__ import annotations`) by looking up
    # sys.modules[cls.__module__], which is None for a module loaded by path.
    sys.modules["asr_metrics_under_test"] = module
    sys.path.insert(0, str(BENCHMARKS))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(BENCHMARKS))
    return module


# --- normalisation -----------------------------------------------------------


def test_lowercases_without_folding_eszett(metrics):
    """str.casefold() maps ß to ss; str.lower() does not. The difference matters.

    Using casefold would silently apply half an umlaut fold regardless of the
    fold_umlauts flag, making that flag untestable and quietly hiding a class of
    real German errors.
    """
    assert metrics.normalize_german("Grüße") == "grüße"
    assert "ss" not in metrics.normalize_german("Grüße")


def test_fold_umlauts_is_opt_in(metrics):
    assert metrics.normalize_german("Grüße Äpfel Öl") == "grüße äpfel öl"
    assert metrics.normalize_german("Grüße Äpfel Öl", fold_umlauts=True) == "gruesse aepfel oel"


def test_composed_and_decomposed_umlauts_are_equal(metrics):
    """NFC first, or a backend emitting decomposed forms scores ~100% WER."""
    composed = "über"                     # U+00FC
    decomposed = "über"             # u + COMBINING DIAERESIS
    assert composed != decomposed
    assert metrics.normalize_german(composed) == metrics.normalize_german(decomposed)
    assert metrics.word_errors(composed, decomposed).errors == 0


def test_hyphens_become_spaces_so_compounds_agree(metrics):
    """German compounds are hyphenated inconsistently even between humans."""
    assert metrics.word_errors("E-Mail-Adresse", "E Mail Adresse").errors == 0
    # but a genuinely different rendering is still counted
    assert metrics.word_errors("E-Mail-Adresse", "Email Adresse").errors > 0


def test_punctuation_and_quote_styles_are_ignored(metrics):
    ref = "„Guten Tag!“, sagte er."
    hyp = '"Guten Tag", sagte er'
    assert metrics.word_errors(ref, hyp).errors == 0


def test_whitespace_is_collapsed(metrics):
    assert metrics.normalize_german("  guten   \t tag \n") == "guten tag"


def test_empty_input_is_empty_not_a_crash(metrics):
    assert metrics.normalize_german("") == ""
    assert metrics.tokenize("") == []
    assert metrics.normalize_german("!!!") == ""


# --- edit distance -----------------------------------------------------------


def test_identical_text_has_no_errors(metrics):
    counts = metrics.word_errors("der schnelle braune fuchs", "der schnelle braune fuchs")
    assert counts.errors == 0
    assert counts.rate == 0.0


def test_counts_substitution_deletion_and_insertion_separately(metrics):
    """An alignment with no cheaper alternative, so the breakdown is unambiguous.

    ref: eins zwei  drei vier
    hyp: eins zwoo  drei vier fuenf
                ^sub                ^ins
    Two operations. Any other two-operation path would have to delete and insert
    in place of the substitution, which costs three.
    """
    counts = metrics.word_errors("eins zwei drei vier", "eins zwoo drei vier fuenf")
    assert counts.substitutions == 1
    assert counts.deletions == 0
    assert counts.insertions == 1
    assert counts.reference_length == 4
    assert counts.rate == 0.5


def test_ties_keep_a_stable_total_even_when_the_breakdown_is_arbitrary(metrics):
    """Equally-minimal alignments exist; only the total is well defined.

    "a b c d" -> "a x d e" costs three either way: three substitutions, or one
    substitution plus a deletion plus an insertion. Asserting a particular
    breakdown here would pin an arbitrary tie-break in the DP, not a property of
    the metric. The error TOTAL is what gets reported and it is stable.
    """
    counts = metrics.word_errors("a b c d", "a x d e")
    assert counts.errors == 3
    assert counts.reference_length == 4
    assert counts.rate == 0.75


def test_pure_deletion(metrics):
    counts = metrics.word_errors("eins zwei drei", "eins drei")
    assert (counts.substitutions, counts.deletions, counts.insertions) == (0, 1, 0)


def test_pure_insertion(metrics):
    counts = metrics.word_errors("eins drei", "eins zwei drei")
    assert (counts.substitutions, counts.deletions, counts.insertions) == (0, 0, 1)


def test_empty_hypothesis_deletes_everything(metrics):
    counts = metrics.word_errors("eins zwei drei", "")
    assert counts.deletions == 3
    assert counts.rate == 1.0


def test_empty_reference_is_finite_not_infinite(metrics):
    """One junk row must not poison a corpus total with inf or ZeroDivisionError."""
    counts = metrics.word_errors("", "unerwartet")
    assert counts.insertions == 1
    assert counts.reference_length == 0
    assert counts.rate == 1.0


def test_both_empty_is_zero(metrics):
    assert metrics.word_errors("", "").rate == 0.0


def test_wer_can_exceed_one(metrics):
    """A hallucinating model inserts more than the reference contains."""
    counts = metrics.word_errors("ja", "ja ja ja ja ja")
    assert counts.rate > 1.0


# --- CER ---------------------------------------------------------------------


def test_cer_is_gentler_than_wer_on_a_long_compound(metrics):
    """One wrong letter in a compound is a whole word error but few characters.

    This is why CER is reported alongside WER for German: WER swings hard on
    compounds, which are exactly where quantisation damage would first appear.
    """
    ref = "Rechtschreibprüfung"
    hyp = "Rechtschreibprufung"
    assert metrics.word_errors(ref, hyp).rate == 1.0
    assert metrics.character_errors(ref, hyp).rate < 0.1


def test_cer_ignores_spacing(metrics):
    assert metrics.character_errors("guten tag", "gutentag").errors == 0


# --- corpus aggregation ------------------------------------------------------


def test_corpus_rate_weights_by_length_not_by_utterance(metrics):
    """Total errors over total tokens — not the mean of per-utterance rates.

    A three-word utterance must not count as much as a sixty-word one; averaging
    rates is the classic way to report a WER that no one else can reproduce.
    """
    pairs = [
        ("a b c", "a b c"),                          # 3 tokens, 0 errors
        ("d e f g h i j k l m", "x x x x x x x x x x"),  # 10 tokens, 10 errors
    ]
    total = metrics.corpus_rate(pairs)
    assert total.reference_length == 13
    assert total.errors == 10
    assert total.rate == pytest.approx(10 / 13)

    mean_of_rates = (0.0 + 1.0) / 2
    assert total.rate != pytest.approx(mean_of_rates)


def test_error_counts_add(metrics):
    a = metrics.ErrorCounts(1, 2, 3, 10)
    b = metrics.ErrorCounts(4, 5, 6, 20)
    total = a + b
    assert (total.substitutions, total.deletions, total.insertions, total.reference_length) == (
        5, 7, 9, 30
    )


# --- paired bootstrap --------------------------------------------------------


def test_identical_systems_are_not_significant(metrics):
    counts = [metrics.ErrorCounts(1, 0, 0, 10) for _ in range(40)]
    result = metrics.paired_bootstrap(counts, list(counts), resamples=500)
    assert result.delta == 0.0
    assert not result.significant
    assert "NO SIGNIFICANT DIFFERENCE" in result.verdict()


def test_a_large_consistent_regression_is_significant(metrics):
    baseline = [metrics.ErrorCounts(1, 0, 0, 20) for _ in range(60)]
    candidate = [metrics.ErrorCounts(8, 0, 0, 20) for _ in range(60)]
    result = metrics.paired_bootstrap(baseline, candidate, resamples=1000)
    assert result.delta > 0
    assert result.significant
    assert result.ci_low > 0
    assert "SIGNIFICANT" in result.verdict()


def test_a_tiny_difference_on_a_small_set_is_not_significant(metrics):
    """The whole point: do not let a 1-utterance difference decide a default."""
    baseline = [metrics.ErrorCounts(2, 0, 0, 20) for _ in range(30)]
    candidate = list(baseline)
    candidate[0] = metrics.ErrorCounts(3, 0, 0, 20)

    result = metrics.paired_bootstrap(baseline, candidate, resamples=1000)
    assert result.delta > 0, "the candidate really is fractionally worse"
    assert not result.significant, "but 30 utterances cannot establish that"
    assert "Do not decide on this" in result.verdict()


def test_bootstrap_is_reproducible_with_a_fixed_seed(metrics):
    baseline = [metrics.ErrorCounts(i % 4, 0, 0, 12) for i in range(50)]
    candidate = [metrics.ErrorCounts((i + 1) % 5, 0, 0, 12) for i in range(50)]
    first = metrics.paired_bootstrap(baseline, candidate, resamples=400, seed=7)
    second = metrics.paired_bootstrap(baseline, candidate, resamples=400, seed=7)
    assert (first.ci_low, first.ci_high) == (second.ci_low, second.ci_high)


def test_bootstrap_is_paired_not_independent(metrics):
    """Per-utterance difficulty must cancel.

    Difficulty varies wildly across utterances but the two systems differ by a
    constant one extra error each. A paired test sees that clearly; an unpaired
    one would mostly measure which utterances got drawn.
    """
    baseline = [metrics.ErrorCounts(i, 0, 0, 40) for i in range(1, 51)]
    candidate = [metrics.ErrorCounts(i + 1, 0, 0, 40) for i in range(1, 51)]
    result = metrics.paired_bootstrap(baseline, candidate, resamples=1000)
    assert result.significant
    assert result.candidate_worse_fraction == 1.0


def test_mismatched_lengths_are_refused(metrics):
    with pytest.raises(ValueError, match="equal lengths"):
        metrics.paired_bootstrap([metrics.ErrorCounts(1, 0, 0, 5)], [])


def test_empty_comparison_is_refused(metrics):
    with pytest.raises(ValueError, match="no utterances"):
        metrics.paired_bootstrap([], [])


def test_confidence_interval_brackets_the_observed_delta(metrics):
    baseline = [metrics.ErrorCounts(i % 3, 0, 0, 15) for i in range(80)]
    candidate = [metrics.ErrorCounts((i % 3) + 1, 0, 0, 15) for i in range(80)]
    result = metrics.paired_bootstrap(baseline, candidate, resamples=1000)
    assert result.ci_low <= result.delta <= result.ci_high


# --- manifest loading --------------------------------------------------------


@pytest.fixture(scope="module")
def runner():
    """Load the CLI module. httpx is only needed when actually transcribing."""
    spec = importlib.util.spec_from_file_location(
        "run_german_eval_under_test", BENCHMARKS / "run_german_eval.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_german_eval_under_test"] = module
    sys.path.insert(0, str(BENCHMARKS))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(BENCHMARKS))
    return module


def test_reads_a_common_voice_tsv(runner, tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "a.mp3").write_bytes(b"")
    (clips / "b.mp3").write_bytes(b"")

    manifest = tmp_path / "validated.tsv"
    manifest.write_text(
        "client_id\tpath\tsentence\tup_votes\n"
        "x\ta.mp3\tGuten Tag\t2\n"
        "y\tb.mp3\tWie geht es dir\t3\n",
        encoding="utf-8",
    )

    rows = runner.load_manifest(manifest)
    assert [text for _, text in rows] == ["Guten Tag", "Wie geht es dir"]
    assert all(path.exists() for path, _ in rows), "clips/ layout must resolve"


def test_skips_common_voice_rows_with_no_sentence(runner, tmp_path):
    (tmp_path / "a.mp3").write_bytes(b"")
    manifest = tmp_path / "m.tsv"
    manifest.write_text("path\tsentence\na.mp3\tGuten Tag\nb.mp3\t\n", encoding="utf-8")
    assert len(runner.load_manifest(manifest)) == 1


def test_reads_jsonl(runner, tmp_path):
    (tmp_path / "a.wav").write_bytes(b"")
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        '{"audio": "a.wav", "text": "Guten Tag"}\n\n', encoding="utf-8"
    )
    rows = runner.load_manifest(manifest)
    assert rows[0][1] == "Guten Tag"


def test_rejects_a_manifest_with_no_usable_columns(runner, tmp_path):
    manifest = tmp_path / "m.csv"
    manifest.write_text("foo,bar\n1,2\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="could not find an audio column"):
        runner.load_manifest(manifest)


def test_missing_manifest_is_a_clear_error(runner, tmp_path):
    with pytest.raises(SystemExit, match="manifest not found"):
        runner.load_manifest(tmp_path / "nope.tsv")
